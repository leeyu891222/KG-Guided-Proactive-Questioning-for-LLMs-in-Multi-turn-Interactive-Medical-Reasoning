# MediQ_test.py

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import time
import torch
#from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import pandas as pd 
import json
from tqdm import tqdm
import argparse
import logging
import datetime
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import networkx as nx
from collections import OrderedDict, defaultdict


try:
    from MediQ_trainer_triattn import Trainer, MediQPreprocessedDataset, collate_fn_mediq_preprocessed, CuiEmbedding
except ImportError as e:
    print(f"導入模組時出錯: {e}")
    print("請確保 MediQ_trainer_triattn.py 在您的 Python 路徑中，並且所有依賴項已安裝。")
    exit()

def load_cui_to_text_map(pickle_path):
    """
    從 pickle 文件中加載 CUI 到可讀文本的映射字典。
    """
    print(f"正在從 {pickle_path} 載入 CUI 詞彙表...")
    try:
        with open(pickle_path, 'rb') as f:
            cui_vocab_data = pickle.load(f)
        
        cui_to_text = {}
        for cui, text_options in cui_vocab_data.items():
            if text_options and isinstance(text_options, list) and len(text_options[0]) > 1:
                # 根據您的要求，使用第一個 prefer text
                cui_to_text[cui] = text_options[0][1]
            else:
                cui_to_text[cui] = cui # 如果沒有文本，則用CUI本身作為回退
        
        print(f"成功創建 {len(cui_to_text)} 條 CUI 到文本的映射。")
        return cui_to_text
    except Exception as e:
        print(f"載入或處理 CUI 詞彙表時出錯: {e}")
        return {}
    
class TestTrainer(Trainer):
    """
    繼承自 Trainer，但重載 forward_per_batch 方法以適應測試需求。
    """
    def forward_per_batch(self, batch):
        """
        重載版本：只進行推理並收集所有高於 collection_threshold 的原始預測，
        不執行路徑取代邏輯。
        """
        # (這個方法的大部分邏輯與原始版本相似，但移除了路徑取代部分)
        input_text_tks_padded = batch['input_text_tks_padded']
         
        hop1_target_cuis_str_batch = batch['hop1_target_cuis']
        hop2_target_cuis_str_batch = batch['hop2_target_cuis']
        intermediate_target_cuis_batch = batch['intermediate_target_cuis']
        input_task_embs_batch = self.encoder(
            input_text_tks_padded['input_ids'].to(self.device),
            input_text_tks_padded['attention_mask'].to(self.device)
        ).pooler_output

        batch_size = input_task_embs_batch.shape[0]
        batch_raw_predictions = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            task_emb_sample = input_task_embs_batch[i].unsqueeze(0)
            known_cuis_str_sample = batch['known_cuis'][i]
            context_emb_sample = self.compute_context_embedding(known_cuis_str_sample)
            if context_emb_sample is None: context_emb_sample = task_emb_sample

            current_cui_str_list_for_hop = known_cuis_str_sample
            prev_iter_state_for_next_hop = None

            for running_k in range(self.k_hops):
                if not current_cui_str_list_for_hop and running_k > 0: break
                
                gt_cuis_str_list_this_hop = []
                if running_k == 0:
                    gt_cuis_str_list_this_hop.extend(hop1_target_cuis_str_batch[i])
                    if self.intermediate:
                        gt_cuis_str_list_this_hop.extend(intermediate_target_cuis_batch[i])
                elif running_k == 1:
                    gt_cuis_str_list_this_hop.extend(hop2_target_cuis_str_batch[i])
                          
                          
                if self.preserve_gt_in_pruning:
                    gt_indices_tensor_for_pruning = self._get_gt_indices_tensor(list(set(gt_cuis_str_list_this_hop)))
                else:
                    gt_indices_tensor_for_pruning = None
                
                all_paths_info_hop, _, next_hop_state_info_for_exploration, stop_flag, _ = self.gmodel.one_iteration(
                    task_emb_sample, current_cui_str_list_for_hop, running_k,
                    context_emb_sample, prev_iter_state_for_next_hop,
                    gt_indices_for_pruning=gt_indices_tensor_for_pruning
                )

                if stop_flag or all_paths_info_hop is None: break

                # --- 【修改核心】只收集，不決策 ---
                path_scores = all_paths_info_hop['scores'].squeeze(-1)
                confident_mask = (path_scores >= self.score_threshold)
                confident_path_indices = torch.where(confident_mask)[0]

                if confident_path_indices.numel() > 0:
                    # 解包所有高於收集閾值的路徑的詳細資訊
                    conf_orig_srcs = all_paths_info_hop['mem_orig_src_idx'][confident_path_indices]
                    conf_first_edges = all_paths_info_hop['mem_first_edge_idx'][confident_path_indices] if all_paths_info_hop['mem_first_edge_idx'] is not None else None
                    conf_hop_srcs = all_paths_info_hop['src_idx'][confident_path_indices] # 當前跳的源 (中間節點 for hop2)
                    conf_hop_edges = all_paths_info_hop['edge_idx'][confident_path_indices] # 當前跳的邊 (Rel2 for hop2)
                    conf_hop_tgts = all_paths_info_hop['tgt_idx'][confident_path_indices]   # 當前跳的目標 (最終目標 for hop2)
                    conf_scores = path_scores[confident_path_indices]

                    for k_path in range(confident_path_indices.numel()):
                        # 將所有找到的、符合最低閾值的路徑資訊全部記錄下來
                        final_t_idx = conf_hop_tgts[k_path].item()
                        target_cui_str = self.rev_cui_vocab_idx_to_str.get(final_t_idx)
                        if target_cui_str:
                            # 構建完整的路徑索引元組
                            full_path_tuple = (
                                conf_orig_srcs[k_path].item(),
                                conf_first_edges[k_path].item() if conf_first_edges is not None else -1,
                                conf_hop_srcs[k_path].item(),
                                conf_hop_edges[k_path].item(),
                                final_t_idx
                            )
                            batch_raw_predictions[i].append({
                                "target_cui": target_cui_str,
                                "score": conf_scores[k_path].item(),
                                "hop": running_k + 1,
                                "path_indices": full_path_tuple
                            })
                
                if next_hop_state_info_for_exploration and \
                   next_hop_state_info_for_exploration['selected_hop_target_idx'] is not None and \
                   next_hop_state_info_for_exploration['selected_hop_target_idx'].numel() > 0 :
                    
                    current_cui_str_list_for_hop = [
                        self.rev_cui_vocab_idx_to_str.get(idx.item()) 
                        for idx in next_hop_state_info_for_exploration['selected_hop_target_idx']
                        if self.rev_cui_vocab_idx_to_str.get(idx.item()) is not None 
                    ]
                    prev_iter_state_for_next_hop = next_hop_state_info_for_exploration 
                else: 
                    break 
        
        # 測試時不計算loss，只返回收集到的原始預測
        return None, batch_raw_predictions

def run_evaluation(trainer, test_loader, cui_to_text_map):
    """
    執行完整的評估流程。
    【最終修正版】: 直接使用 Trainer 的內部決策邏輯，在每個閾值下重新推理。
    """
    # 將模型切換到評估模式
    trainer.mode = 'eval'
    trainer.encoder.eval()
    trainer.gmodel.eval()

    print("\n階段 1 & 2: 正在對不同閾值進行推理並搜索最佳結果...")
    
    best_f1_final = -1.0
    best_threshold = -1.0
    best_metrics = {}
    best_predictions_for_report = None # 儲存最佳F1對應的、包含詳細資訊的預測列表

    # 遍歷所有待測試的閾值
    thresholds_to_check = np.arange(0.5, 1.0, 0.05)
    for threshold in tqdm(thresholds_to_check, desc="搜索最佳閾值中"):
        
        # 對於每個閾值，都需要重新收集預測和GT
        all_predictions_at_current_threshold = []
        all_gts_for_metric = {'hop1_target_cuis': [], 'hop2_target_cuis': []}

        with torch.no_grad():
            for batch in test_loader:
                if batch is None: continue
                
                # --- 【核心】直接調用原始 Trainer 的 forward_per_batch ---
                # 將當前搜索的 threshold 作為內部決策的依據傳入
                # 該方法內部會完成篩選和路徑取代的所有邏輯
                trainer.score_threshold = threshold
                _, batch_detailed_predictions = trainer.forward_per_batch(batch)
                
                # 收集這個批次的詳細預測結果
                all_predictions_at_current_threshold.extend(batch_detailed_predictions)
                
                # 收集這個批次的GT
                all_gts_for_metric['hop1_target_cuis'].extend(batch['hop1_target_cuis'])
                all_gts_for_metric['hop2_target_cuis'].extend(batch['hop2_target_cuis'])

        # --- 在收集完所有數據後，一次性計算當前閾值的指標 ---
        current_metrics = trainer.measure_accuracy(all_predictions_at_current_threshold, all_gts_for_metric)
        
        # 更新最佳結果
        if current_metrics['final']['f1'] > best_f1_final:
            best_f1_final = current_metrics['final']['f1']
            best_threshold = threshold
            best_metrics = current_metrics
            # 儲存產生最佳結果時的【詳細預測列表】，用於後續錯誤分析
            best_predictions_for_report = all_predictions_at_current_threshold

    print(f"\n最佳閾值搜索完成！最佳 F1@Final: {best_f1_final:.4f} (在閾值 = {best_threshold:.2f} 時取得)")

    # --- 3. 錯誤分析 ---
    print(f"\n階段 3: 正在使用最佳閾值 {best_threshold:.2f} 進行錯誤分析...")
    error_cases = []
    correct_cases_preview = []
    
    if best_predictions_for_report is not None:
        # 使用索引遍歷，從 all_samples_results 獲取 case_id 和 GT
        # 注意：我們需要一個方法在測試開始前先遍歷一次dataloader來獲取 all_samples_results 的元數據
        # 這裡為了簡化，假設 all_samples_results 已在外部定義並包含了GT和case_id
        # 一個更簡單的做法是在這裡重新準備GT列表
        all_gts = []
        all_case_ids = []
        # 這部分資訊可以從 test_loader.dataset.samples 中獲取
        for sample in test_loader.dataset.samples:
            gt1 = set(sample['hop1_target_cuis'])
            gt2 = set(sample['hop2_target_cuis'])
            all_gts.append(gt1.union(gt2))
            all_case_ids.append(sample['case_id'])

        for i in range(len(best_predictions_for_report)):
            case_id = all_case_ids[i]
            gt_combined = all_gts[i]
            
            # 從儲存的最佳預測中獲取CUI集合
            final_preds_set = {p['target_cui'] for p in best_predictions_for_report[i]}
            
            if final_preds_set != gt_combined:
                false_positives = final_preds_set - gt_combined
                false_negatives = gt_combined - final_preds_set
                
                error_cases.append({
                    "case_id": case_id,
                    "ground_truth_combined": {
                        "cuis": sorted(list(gt_combined)),
                        "texts": sorted([cui_to_text_map.get(c, c) for c in gt_combined])
                    },
                    "predictions_final": {
                        "cuis": sorted(list(final_preds_set)),
                        "texts": sorted([cui_to_text_map.get(c, c) for c in final_preds_set])
                    },
                    "error_details": {
                        "false_positives": [{"cui": c, "text": cui_to_text_map.get(c, c)} for c in sorted(list(false_positives))],
                        "false_negatives": [{"cui": c, "text": cui_to_text_map.get(c, c)} for c in sorted(list(false_negatives))]
                    }
                })
            elif len(correct_cases_preview) < 5:
                correct_cases_preview.append({
                    "case_id": case_id,
                    "ground_truth_combined": {
                        "cuis": sorted(list(gt_combined)),
                        "texts": sorted([cui_to_text_map.get(c, c) for c in gt_combined])
                    },
                    "predictions_final": {
                        "cuis": sorted(list(final_preds_set)),
                        "texts": sorted([cui_to_text_map.get(c, c) for c in final_preds_set])
                    }
                })
    else:
        print("警告：未能找到最佳預測結果，無法進行錯誤分析。")

    # --- 4. 準備最終報告 ---
    final_report = {
        "test_file": os.path.basename(getattr(test_loader.dataset, 'preprocessed_file_path', 'N/A')),
        "model_checkpoint": trainer.save_model_path,
        "evaluation_summary": {
            "optimal_threshold": round(best_threshold, 4),
            "metrics_at_optimal_threshold": best_metrics
        },
        "error_analysis": error_cases,
        "correct_cases_preview": correct_cases_preview
    }
    
    return final_report


def main(args):
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    print(f"Using device: {device}")

    # 載入 CUI 到文本的映射
    cui_to_text_map = load_cui_to_text_map(args.cui_vocab_path)
    if not cui_to_text_map: return

    # 載入一個佔位的 encoder 和 graph 以實例化 Trainer
    # Trainer 內部會創建自己的模型
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    base_encoder_model = AutoModel.from_pretrained(args.tokenizer_path)
    g_nx_loaded = pickle.load(open(args.graph_path, "rb"))
    cui_embedding_lookup_obj = CuiEmbedding(args.cui_embedding_path, device=device)
    _nodes_for_vocab = sorted(list(g_nx_loaded.nodes()))
    cui_vocab_for_trainer = {cui_str: i for i, cui_str in enumerate(_nodes_for_vocab)}

    # 實例化 Trainer (配置與訓練時盡量保持一致，除了訓練相關參數)
    trainer = Trainer(
        tokenizer=tokenizer,
        encoder=base_encoder_model,
        g_nx=g_nx_loaded,
        cui_embedding_lookup=cui_embedding_lookup_obj,
        relation_embedding_filepath='drknows/relation_sapbert_embeddings.pkl',
        hdim=base_encoder_model.config.hidden_size,
        nums_of_head=3,
        cui_vocab_str_to_idx=cui_vocab_for_trainer,
        top_n=8,
        device=device,
        nums_of_epochs=1, 
        LR=1e-5,
        loss_type="FOCAL",
        focal_alpha=0.95,
        focal_gamma=2.0,
        cui_weights_dict=None,
        contrastive_learning=False,
        intermediate=True,
        score_threshold=args.collection_threshold,
        finetune_encoder=False,        
        use_soft_labels=False,         
        soft_label_threshold=0.8,     
        preserve_gt_in_pruning=True,         
        save_model_path=None,
        gnn_update=True, 
        path_encoder_type="Transformer",
        path_ranker_type="Flat",
        gnn_type="GAT",
        gin_hidden_dim=base_encoder_model.config.hidden_size,
        gin_num_layers=2,
        early_stopping_patience=3,
        early_stopping_metric='val_loss',
        early_stopping_delta=0.0001,
        lambda_triplet=0.01,
        analyze_pruning=False,
        scheduler_type="warmup",
        warmup_steps=2000
    )
    
    # 載入訓練好的模型權重
    print(f"正在從 {args.gmodel_path} 載入 GraphModel 權重...")
    trainer.gmodel.load_state_dict(torch.load(args.gmodel_path, map_location=device))
    print(f"正在從 {args.encoder_path} 載入 Encoder 權重...")
    trainer.encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
    print("模型權重載入完成。")

    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)

    # 遍歷測試文件夾中的所有 .jsonl 文件
    test_files_found = [f.path for f in os.scandir(args.test_data_dir) if f.name.endswith('.jsonl')]
    print(f"在 {args.test_data_dir} 中找到 {len(test_files_found)} 個測試文件。")

    for test_file_path in test_files_found:
        print(f"\n{'='*20} 開始測試文件: {os.path.basename(test_file_path)} {'='*20}")
        
        # 為每個文件創建 Dataset 和 DataLoader
        test_dataset = MediQPreprocessedDataset(test_file_path)
        if len(test_dataset) == 0:
            print(f"警告：測試文件 {test_file_path} 為空或無法載入樣本，跳過。")
            continue
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_mediq_preprocessed, num_workers=6, pin_memory=True)

        # 執行評估
        final_report_data = run_evaluation(trainer, test_loader, cui_to_text_map)

        # 生成並保存報告文件
        report_filename = os.path.basename(test_file_path).replace('.jsonl', '_report.json')
        report_output_path = os.path.join(args.output_dir, report_filename)
        
        with open(report_output_path, 'w', encoding='utf-8') as f:
            json.dump(final_report_data, f, indent=4, ensure_ascii=False)
        
        print(f"測試報告已生成: {report_output_path}")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="為 MediQ 任務評估訓練好的圖譜推理模型")
    parser.add_argument("--test_data_dir", type=str, default='./Evaluate', help="包含多個預處理後 .jsonl 測試集的資料夾路徑。")
    parser.add_argument("--gmodel_path", type=str, default='saved_models_mediq/0622_complete_RGAT_noCL_nodiff_intloss/gmodel_mediq_best.pth', help="已訓練好的 GraphModel 權重檔案 (.pth) 路徑。")
    parser.add_argument("--encoder_path", type=str, default='saved_models_mediq/0622_complete_RGAT_noCL_nodiff_intloss/encoder.pth', help="已訓練好的 Encoder (SapBERT) 權重檔案 (.pth) 路徑。")
    parser.add_argument("--output_dir", type=str, default='./Evaluate', help="儲存評估報告的資料夾路徑。")
    parser.add_argument("--cui_vocab_path", type=str, default='./drknows/sm_t047_cui_aui_eng.pkl', help="CUI 到文本映射的 pickle 檔案路徑。")
    parser.add_argument("--graph_path", type=str, default='./drknows/SNOMED_CUI_MAJID_Graph_wSelf.pkl', help="用於模型初始化的 NetworkX 圖譜 pickle 檔案路徑。")
    parser.add_argument("--cui_embedding_path", type=str, default='./drknows/GraphModel_SNOMED_CUI_Embedding.pkl', help="CUI 嵌入的 pickle 檔案路徑。")
    parser.add_argument("--tokenizer_path", type=str, default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext", help="Tokenizer 的路徑或Hugging Face名稱。")
    parser.add_argument("--batch_size", type=int, default=2, help="進行推理時的批次大小。")
    parser.add_argument("--collection_threshold", type=float, default=0.7, help="收集潛在預測時使用的初始低閾值。")
    parser.add_argument("--gnn_type", type=str, default="GAT", help="載入的模型使用的GNN類型 ('GIN', 'GAT', 'Stack')。必須與訓練時一致。")
    parser.add_argument("--num_heads", type=int, default=3, help="GAT的頭數，必須與訓練時一致。")
    parser.add_argument("--force_cpu", action='store_true', help="強制使用CPU進行測試。")
    args = parser.parse_args() 
    main(args)