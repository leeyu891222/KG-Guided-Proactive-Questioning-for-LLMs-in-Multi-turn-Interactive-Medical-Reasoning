
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

# ==============================================================================
# 導入你現有的訓練腳本中的 Trainer 類別和其依賴項
# 這裡我們假設你的 MediQ_trainer_triattn.py 已經被組織成一個可導入的模組
# 如果不是，你需要將 Trainer 類別和其依賴的所有類別
# (如 GraphModel, MediQAnnotatedDataset 等) 放到這個文件中或一個可導入的模組中
# ==============================================================================
try:
    # 理想情況下，你可以直接導入
    from MediQ_trainer_triattn import Trainer, MediQAnnotatedDataset, CuiEmbedding 
except ImportError:
    print("警告：無法從 MediQ_trainer_triattn.py 導入 Trainer。")
    print("將使用佔位符類別，請確保將真實的 Trainer 類別及其依賴項放在此處。")
    # 如果無法導入，你需要將 MediQ_trainer_triattn.py 中的完整 Trainer 類別定義複製到這裡
    from MediQ_trainer_triattn import Trainer 


class KGReasoner(Trainer):
    """
    一個封裝了 KG 推理能力的預測器。
    它繼承自 Trainer，以複用模型加載和前向傳播邏輯，但在評估模式下運行。
    """
    def __init__(self, cui_to_text_map_path: str, **kwargs):
        """
        初始化 KGReasoner。
        
        Args:
            cui_to_text_map_path (str): CUI 編號到可讀文本的映射文件路徑 (JSONL 格式)。
            **kwargs: 所有傳遞給父類 Trainer 的初始化參數，例如：
                      tokenizer, encoder, g_nx, cui_embedding_lookup, hdim, 等等。
        """
        print("初始化 KGReasoner...")
        
        # 1. 調用父類 Trainer 的初始化方法，加載所有模型和資源
        super().__init__(**kwargs)
        
        # 2. 將模型設置為評估模式
        self.encoder.eval()
        self.gmodel.eval()
        print("模型已設置為評估模式 (eval mode)。")

        # 3. 加載 CUI 到文本的映射文件
        self.cui_to_text_map = self._load_cui_to_text_map(cui_to_text_map_path)
        
        # 4. 提取索引到關係名稱的映射，方便後處理
        self.idx_to_relation_map = self.gmodel.g_tensorized.get('idx_to_edge', {})

    def _load_cui_to_text_map(self, file_path: str) -> dict:
        """
        加載 CUI 到可讀文本的映射文件。
        此方法已根據您的要求更新，使用 pickle 加載 .pkl 文件。
        """
        print(f"正在從 {file_path} (使用 pickle) 加載 CUI-to-Text 映射...")
        mapping = {}
        try:
            # 根據您的要求，使用 'rb' 模式和 'latin1' 編碼來加載 pickle 文件
            with open(file_path, 'rb') as f:
                drknows_vocab = pickle.load(f, encoding='latin1')

            # 遍歷加載的字典
            for cui, names_list in drknows_vocab.items():
                # 數據結構是 {'CUI': [['AUI', 'Name'], ['AUI', 'Name'], ...]}
                if names_list and isinstance(names_list, list) and len(names_list[0]) > 1:
                    # 使用第一個 preferred text
                    mapping[cui] = names_list[0][1]
                else:
                    # 如果格式不符，則使用 CUI 本身作為備用
                    mapping[cui] = cui

            print(f"成功加載 {len(mapping)} 條 CUI-to-Text 映射。")
            return mapping
            
        except FileNotFoundError:
            print(f"錯誤: 找不到 CUI-to-Text 映射文件 {file_path}。返回空字典。")
            return {}
        except Exception as e:
            print(f"加載或解析 CUI-to-Text .pkl 文件時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _post_process_predictions(self, detailed_predictions: list):
        """
        對 forward_per_batch 的輸出進行後處理，生成可讀的路徑描述。
        """
        if not detailed_predictions:
            return [], 0.0

        processed_paths = []
        top_score = 0.0

        for pred in detailed_predictions:
            path_tuple = pred['path_indices']
            score = pred['score']
            hop = pred['hop']
            
            top_score = max(top_score, score)

            # 將索引轉換回 CUI 和關係名稱
            orig_src_cui = self.rev_cui_vocab_idx_to_str.get(path_tuple[0])
            rel1_name = self.idx_to_relation_map.get(path_tuple[1], "未知關係1")
            
            # 將 CUI 轉換為可讀文本
            orig_src_text = self.cui_to_text_map.get(orig_src_cui, orig_src_cui)

            path_text = ""
            if hop == 1:
                target_cui = self.rev_cui_vocab_idx_to_str.get(path_tuple[2])
                target_text = self.cui_to_text_map.get(target_cui, target_cui)
                path_text = f"'{orig_src_text}' --[{rel1_name}]--> '{target_text}'"
            elif hop == 2:
                inter_cui = self.rev_cui_vocab_idx_to_str.get(path_tuple[2])
                rel2_name = self.idx_to_relation_map.get(path_tuple[3], "未知關係2")
                final_target_cui = self.rev_cui_vocab_idx_to_str.get(path_tuple[4])
                
                inter_text = self.cui_to_text_map.get(inter_cui, inter_cui)
                final_target_text = self.cui_to_text_map.get(final_target_cui, final_target_cui)
                
                path_text = f"'{orig_src_text}' --[{rel1_name}]--> '{inter_text}' --[{rel2_name}]--> '{final_target_text}'"
            
            processed_paths.append({
                #"target_cui": self.rev_cui_vocab_idx_to_str.get(pred['target_cui']),
                "score": score,
                "hop": hop,
                "path_text": path_text
            })

        

        return processed_paths, top_score

    def predict(self, known_cuis: list, mock_text: str = "Patient information."):
        """
        接收一個已知 CUI 列表，返回預測的路徑、最高分數和分析文本。
        
        Args:
            known_cuis (list): 當前已知的 CUI 字符串列表。
            mock_text (str): 用於生成任務嵌入的模擬文本。
            
        Returns:
            tuple: (processed_paths, top_score, path_analysis_text)
        """
        print(f"KG-Reasoner 接收到 {len(known_cuis)} 個已知 CUIs 進行預測: {known_cuis}")
        
        # 1. 準備一個 dummy batch
        # forward_per_batch 需要一個完整的批次結構
        tokenized_input = self.tokenizer(mock_text,
                                         truncation=True,
                                         padding="max_length",
                                         max_length=512,
                                         return_tensors="pt")
        
        dummy_batch = {
            "case_id": ["dummy_case"],
            "input_text_tks_padded": {
                "input_ids": tokenized_input['input_ids'].to(self.device),
                "attention_mask": tokenized_input['attention_mask'].to(self.device)
            },
            "known_cuis": [known_cuis],
            # GTs 在預測時為空
            "hop1_target_cuis": [[]],
            "hop2_target_cuis": [[]],
            "intermediate_target_cuis": [[]]
        }

        # 2. 在 no_grad 模式下調用 forward_per_batch
        with torch.no_grad():
            _, batch_detailed_predictions = self.forward_per_batch(dummy_batch)

        # 3. 提取單個樣本的預測結果並進行後處理
        # 因為我們的批次大小為 1，所以只取第一個結果
        predictions_for_sample = batch_detailed_predictions[0]
        
        processed_paths, top_score = self._post_process_predictions(predictions_for_sample)
        
        print(f"KG-Reasoner 預測完成。最高分: {top_score:.4f}")
        return processed_paths, top_score

# ==============================================================================
# Main Block - 用於演示和測試 KGReasoner
# ==============================================================================
if __name__ == '__main__':
    
    print("開始演示 KGReasoner 的使用...")

    # --- 1. 定義所有必要資源的路徑 (請替換為你的真實路徑) ---
    CUI_TO_TEXT_MAP_FILE = "./drknows/sm_t047_cui_aui_eng.pkl"
    GRAPH_NX_FILE = "./drknows/SNOMED_CUI_MAJID_Graph_wSelf.pkl"
    CUI_EMBEDDING_FILE = "./drknows/GraphModel_SNOMED_CUI_Embedding.pkl"
    TOKENIZER_PATH = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    
    # --- 你的預訓練模型權重路徑 ---
    SAVED_GMODEL_PATH = "./saved_models_mediq/gmodel_mediq_best.pth"
    SAVED_ENCODER_PATH = "./saved_models_mediq/encoder.pth"
    
    # 檢查文件是否存在
    required_files = [CUI_TO_TEXT_MAP_FILE, GRAPH_NX_FILE, CUI_EMBEDDING_FILE, SAVED_GMODEL_PATH, SAVED_ENCODER_PATH]
    for f_path in required_files:
        if not os.path.exists(f_path):
            print(f"錯誤：找不到必要的資源文件 -> {f_path}")
            print("請確保所有路徑都已正確設置，並將此 main block 中的佔位符替換掉。")
            exit()

    # --- 2. 準備 Trainer 初始化所需的參數 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("正在加載基礎資源 (tokenizer, encoder, graph)...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    base_encoder_model = AutoModel.from_pretrained(TOKENIZER_PATH)
    g_nx_loaded = pickle.load(open(GRAPH_NX_FILE, "rb"))
    cui_embedding_lookup_obj = CuiEmbedding(CUI_EMBEDDING_FILE, device=device)
    _nodes_for_vocab = sorted(list(g_nx_loaded.nodes()))
    cui_vocab_for_trainer = {cui_str: i for i, cui_str in enumerate(_nodes_for_vocab)}
    
    # Trainer 的超參數 (應與你訓練時使用的參數一致)
    hdim = base_encoder_model.config.hidden_size
    
    # 將所有參數打包到一個字典中
    trainer_kwargs = {
        'tokenizer': tokenizer,
        'encoder': base_encoder_model,
        'g_nx': g_nx_loaded,
        'cui_embedding_lookup': cui_embedding_lookup_obj,
        'hdim': hdim,
        'nums_of_head': 3,
        'cui_vocab_str_to_idx': cui_vocab_for_trainer,
        'top_n': 8,
        'device': device,
        'nums_of_epochs': 1, # 對預測不重要
        'LR': 1e-5,  # 對預測不重要
        'gnn_update': True, 
        'path_encoder_type': "Transformer",
        'path_ranker_type': "Flat",
        'gnn_type': 'GAT',   # 應與你訓練的模型匹配
        'score_threshold': 0.7,
        'gin_hidden_dim': hdim,
        'gin_num_layers': 2,
    }
    
    # --- 3. 實例化 KGReasoner ---
    try:
        kg_reasoner_instance = KGReasoner(
            cui_to_text_map_path=CUI_TO_TEXT_MAP_FILE,
            **trainer_kwargs
        )
        
        # 加載預訓練的權重
        kg_reasoner_instance.gmodel.load_state_dict(torch.load(SAVED_GMODEL_PATH, map_location=device))
        kg_reasoner_instance.encoder.load_state_dict(torch.load(SAVED_ENCODER_PATH, map_location=device))
        print("成功加載預訓練模型權重。")

    except Exception as e:
        print(f"實例化或加載 KGReasoner 時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # --- 4. 進行一次模擬預測 ---
    sample_known_cuis = ['C0015967', 'C0010066'] # 假設已知 'Fever' 和 'Cough'
    
    print(f"\n--- 開始使用已知 CUIs 進行預測: {sample_known_cuis} ---")
    
    predicted_paths, top_score, path_analysis_text = kg_reasoner_instance.predict(sample_known_cuis)
    
    print("\n--- 預測結果 ---")
    print(f"最高分數: {top_score}")
    print(f"路徑分析摘要: {path_analysis_text}")
    print("\n詳細路徑列表:")
    for path in predicted_paths:
        print(f"  - Hop: {path['hop']}, Score: {path['score']:.4f}, Path: {path['path_text']}")