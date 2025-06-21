# Required imports (ensure these are at the top of your file)

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


# Placeholder for a tokenizer if not loaded globally
# tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext") # Or your model path
# Try to import torch_scatter, if not available, print a warning
try:
    from torch_scatter import scatter_add, scatter_max, scatter_softmax
    TORCH_SCATTER_AVAILABLE = True
    print("torch_scatter is available.")
except ImportError:
    TORCH_SCATTER_AVAILABLE = False
    print("Warning: torch_scatter not found. GIN aggregation will be less efficient or simplified.")
    # Define placeholder functions if torch_scatter is not available for basic functionality
    def scatter_add(src, index, dim_size):
        # Naive scatter_add for CPU, very inefficient for large tensors on GPU
        # This is just a placeholder for the concept.
        # For actual use without torch_scatter, a more optimized loop or other PyTorch ops are needed.
        out = torch.zeros(dim_size, src.size(1), dtype=src.dtype, device=src.device)
        # This is a simple loop, not optimized.
        for i in range(src.size(0)):
            out[index[i]] += src[i]
        return out
    
class MediQAnnotatedDataset(Dataset):
    def __init__(self, annotation_file_path, tokenizer, random_seed=2023, min_facts_for_sampling=2):
        
        print(f"載入並預處理標註數據: {annotation_file_path}")
        self.min_facts_for_sampling = max(2, min_facts_for_sampling)
        self.tokenizer = tokenizer
        self.random = random.Random(random_seed)
        self.valid_training_samples = [] 

        try:
            with open(annotation_file_path, 'r', encoding='utf-8') as f:
                all_annotations = json.load(f)
        except FileNotFoundError:
            print(f"錯誤：找不到標註文件 {annotation_file_path}。")
            raise
        except Exception as e:
            print(f"載入標註文件時發生錯誤: {e}")
            raise

        num_cases_before_filter = len(all_annotations)
        num_cases_after_min_facts_filter = 0

        for case_id, data in all_annotations.items():
          
            atomic_facts = data.get("atomic_facts", [])
            facts_cuis = data.get("facts_cuis", [])
            paths_between_facts = data.get("paths_between_facts", {})
            num_facts = len(atomic_facts)

            if num_facts < self.min_facts_for_sampling or len(facts_cuis) != num_facts:
                continue
            num_cases_after_min_facts_filter += 1

            for i in range(num_facts):
                for j in range(num_facts):
                    if i == j:
                        continue
                    
                    path_key = f"{i}_{j}"
                    if path_key in paths_between_facts and paths_between_facts[path_key]:
                        if facts_cuis[j]:
                            has_valid_target_in_path = False
                            for path_data in paths_between_facts[path_key]:
                                if not path_data or not isinstance(path_data, list): continue
                                target_cui_in_path = path_data[-1]
                                if isinstance(target_cui_in_path, str) and target_cui_in_path.startswith('C') and \
                                   target_cui_in_path in facts_cuis[j]:
                                    has_valid_target_in_path = True
                                    break
                            if has_valid_target_in_path:
                                self.valid_training_samples.append({
                                    "case_id": case_id,
                                    "guaranteed_known_idx": i,
                                    "guaranteed_unknown_idx": j,
                                })
        
        self.all_annotations_data = all_annotations

        print(f"原始案例數: {num_cases_before_filter}")
        print(f"通過最小事實數 ({self.min_facts_for_sampling}) 和 CUI 長度檢查的案例數: {num_cases_after_min_facts_filter}")
        print(f"構建的有效 (known_idx, unknown_idx) 訓練樣本對數量: {len(self.valid_training_samples)}")

        if not self.valid_training_samples:
            print("警告: 預處理後沒有找到任何有效的訓練樣本對。請檢查數據和 `paths_between_facts` 的內容。")

    def __len__(self):
        return len(self.valid_training_samples)

    def __getitem__(self, index):
        if index >= len(self.valid_training_samples):
            raise IndexError("Index out of bounds for valid_training_samples")
            
        sample_info = self.valid_training_samples[index]
        case_id = sample_info["case_id"]
        guaranteed_known_idx = sample_info["guaranteed_known_idx"]
        guaranteed_unknown_idx = sample_info["guaranteed_unknown_idx"]
        
        data = self.all_annotations_data[case_id]
        atomic_facts = data["atomic_facts"]
        facts_cuis = data["facts_cuis"]
        paths_between_facts = data.get("paths_between_facts", {})
        num_facts = len(atomic_facts)

        known_indices = {guaranteed_known_idx}
        unknown_indices = {guaranteed_unknown_idx}

        remaining_indices = list(set(range(num_facts)) - known_indices - unknown_indices)
        self.random.shuffle(remaining_indices)

        if remaining_indices:
            num_additional_known = self.random.randint(0, len(remaining_indices))
            known_indices.update(remaining_indices[:num_additional_known])
            unknown_indices.update(remaining_indices[num_additional_known:])

        known_texts_list = [atomic_facts[i] for i in sorted(list(known_indices))]
        known_texts_combined = " ".join(known_texts_list) if known_texts_list else "N/A"
        
        input_text_tks = self.tokenizer(known_texts_combined,
                                        truncation=True, padding="max_length",
                                        max_length=512, return_tensors="pt")
        input_text_tks = {k: v.squeeze(0) for k, v in input_text_tks.items()}

        known_cuis_list_flat = []
        for i in known_indices:
            current_fact_cuis = facts_cuis[i]
            if isinstance(current_fact_cuis, list):
                known_cuis_list_flat.extend(current_fact_cuis)
        known_cuis_list_unique = list(set(known_cuis_list_flat))
        
        # --- 【修改點 1】將已知CUI列表轉換為集合，以提高查詢效率 ---
        known_cuis_set_for_filtering = set(known_cuis_list_unique)

        hop1_target_cuis_set = set()
        hop2_target_cuis_set = set()
        intermediate_target_cuis_set = set()
        
        for k_idx in known_indices:
            for u_idx in unknown_indices:
                path_key_optional = f"{k_idx}_{u_idx}"
                if path_key_optional in paths_between_facts:
                    for path_data in paths_between_facts[path_key_optional]:
                        if not path_data or not isinstance(path_data, list) or len(path_data) < 1: continue
                        
                        path_len = len(path_data)
                        target_cui_in_path = path_data[-1]
                        is_valid_cui_str = isinstance(target_cui_in_path, str) and target_cui_in_path.startswith('C')
                        
                        if is_valid_cui_str and isinstance(facts_cuis[u_idx], list) and \
                           target_cui_in_path in facts_cuis[u_idx]:
                            
                            # --- 【修改點 2】在添加GT之前，檢查最終目標CUI是否已經是已知的 ---
                            if target_cui_in_path in known_cuis_set_for_filtering:
                                # 如果最終目標CUI已經在已知集合中，則這不是一條有效的探索路徑GT，跳過。
                                continue
                            # --- 修改結束 ---

                            if path_len == 3: # 1-hop path
                                hop1_target_cuis_set.add(target_cui_in_path)
                            elif path_len == 5: # 2-hop path
                                intermediate_cui = path_data[2]
                                if isinstance(intermediate_cui, str) and intermediate_cui.startswith('C'):
                                    # 根據您的要求，中間節點可以是已知的，所以這裡不過濾。
                                    intermediate_target_cuis_set.add(intermediate_cui)
                                
                                # 2跳路徑的最終目標已經在上面被過濾過了。
                                hop2_target_cuis_set.add(target_cui_in_path)
        
        # (此處的返回邏輯保持不變)
        if not hop1_target_cuis_set and not hop2_target_cuis_set:
            return None

        return {
            "case_id": case_id,
            "known_indices": sorted(list(known_indices)),
            "unknown_indices": sorted(list(unknown_indices)),
            "input_text_tks": input_text_tks,
            "known_cuis": known_cuis_list_unique,
            "hop1_target_cuis": list(hop1_target_cuis_set),
            "hop2_target_cuis": list(hop2_target_cuis_set),
            "intermediate_target_cuis": list(intermediate_target_cuis_set),
        }


def collate_fn_mediq_paths(batch):
    valid_items_in_batch = [item for item in batch if item is not None]
    if not valid_items_in_batch:
        return None
    
    # 提取所有鍵
    case_ids = [item['case_id'] for item in valid_items_in_batch]
    input_ids = [item['input_text_tks']['input_ids'] for item in valid_items_in_batch]
    attention_mask = [item['input_text_tks']['attention_mask'] for item in valid_items_in_batch]
    known_cuis = [item['known_cuis'] for item in valid_items_in_batch]
    hop1_target_cuis = [item['hop1_target_cuis'] for item in valid_items_in_batch]
    hop2_target_cuis = [item['hop2_target_cuis'] for item in valid_items_in_batch]
    intermediate_target_cuis = [item['intermediate_target_cuis'] for item in valid_items_in_batch] # ## ADDED

    # 填充
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        "case_id": case_ids,
        "input_text_tks_padded": {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded
        },
        "known_cuis": known_cuis,
        "hop1_target_cuis": hop1_target_cuis,
        "hop2_target_cuis": hop2_target_cuis,
        "intermediate_target_cuis": intermediate_target_cuis, # ## ADDED
    }
    
    
    
class MediQPreprocessedDataset(Dataset):
    def __init__(self, preprocessed_file_path, tokenizer_for_padding_info=None):
        """
        初始化 Dataset.
        Args:
            preprocessed_file_path (str): 預處理後的 .jsonl 檔案路徑。
            tokenizer_for_padding_info (optional): Tokenizer實例，主要用於獲取 pad_token_id。
                                                    如果預處理時已padding到固定長度，則可能不需要。
                                                    但如果collate_fn需要動態padding，知道pad_id有時有用。
                                                    在這個實現中，我們假設預處理時已經padding。
        """
        print(f"開始從預處理文件載入數據: {preprocessed_file_path}")
        self.preprocessed_file_path = preprocessed_file_path
        self.samples = []
        try:
            with open(preprocessed_file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="正在載入預處理樣本"):
                    try:
                        self.samples.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        print(f"警告: 無法解析行: {line.strip()}")
        except FileNotFoundError:
            print(f"錯誤：找不到預處理文件 {preprocessed_file_path}。")
            raise
        except Exception as e:
            print(f"載入預處理文件時發生錯誤: {e}")
            raise
        
        if not self.samples:
            print("警告: Dataset 初始化後為空。請檢查預處理文件。")
        else:
            print(f"成功載入 {len(self.samples)} 個預處理樣本。")
            # 檢查第一個樣本的結構 (可選)
            # print(f"第一個樣本的鍵: {self.samples[0].keys()}")

        # 如果您的 collate_fn 需要 tokenizer 的 pad_token_id，可以在這裡保存
        # self.pad_token_id = tokenizer_for_padding_info.pad_token_id if tokenizer_for_padding_info else 0


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        獲取一個預處理好的樣本並將其轉換為 PyTorch 張量。
        """
        sample_data = self.samples[index]
        
        # 從 sample_data 中提取各個部分
        # 預處理腳本應該已經將 tokenized_input_ids 和 tokenized_attention_mask 
        # 存儲為整數列表，並且已經 padding 到了固定長度 (例如 512)
        
        input_ids = torch.tensor(sample_data["tokenized_input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(sample_data["tokenized_attention_mask"], dtype=torch.long)
        
        # known_cuis, hop1_target_cuis, hop2_target_cuis, intermediate_target_cuis
        # 這些是 CUI 字符串列表，在 Trainer 的 forward_per_batch 中會被使用
        # Dataset 層面不需要將它們轉換為索引，因為 Trainer 會處理
        known_cuis = sample_data["known_cuis"]
        hop1_target_cuis = sample_data["hop1_target_cuis"]
        hop2_target_cuis = sample_data["hop2_target_cuis"]
        intermediate_target_cuis = sample_data["intermediate_target_cuis"]
        
        # 如果您在預處理時還保存了 case_id 或其他元數據，也可以在這裡一併返回
        case_id = sample_data.get("case_id", f"sample_{index}") 
        # known_indices = sample_data.get("known_indices", []) # 如果需要
        # unknown_indices = sample_data.get("unknown_indices", []) # 如果需要

        return {
            "case_id": case_id,
            # "known_indices": known_indices, # 可選
            # "unknown_indices": unknown_indices, # 可選
            "input_text_tks_padded": { # collate_fn 將直接使用這些已是Tensor的數據
                "input_ids": input_ids,
                "attention_mask": attention_mask
            },
            "known_cuis": known_cuis, # List[str]
            "hop1_target_cuis": hop1_target_cuis, # List[str]
            "hop2_target_cuis": hop2_target_cuis, # List[str]
            "intermediate_target_cuis": intermediate_target_cuis # List[str]
        }

# --- 您的 collate_fn_mediq_paths 函數 (用於 DataLoader) ---
# 這個 collate_fn 現在接收的批次中的每個 item，其 'input_text_tks_padded' 
# 已經是包含正確padding的張量了。所以 collate_fn 主要是將這些張量 stack 起來。
def collate_fn_mediq_preprocessed(batch):
    # batch 是一個列表，其中每個元素是 __getitem__ 返回的字典
    if not batch: # 如果批次為空 (例如，所有樣本都被過濾了，儘管在這個 Dataset 中不應該發生)
        return None

    # 從批次中的第一個有效樣本檢查鍵 (假設所有樣本結構一致)
    # valid_items_in_batch = [item for item in batch if item is not None] # 理論上不應有None
    # if not valid_items_in_batch: return None
    # first_item = valid_items_in_batch[0]
    first_item = batch[0]


    collated_batch = {}
    keys_to_stack = [] # 需要堆疊成批次張量的鍵
    keys_as_list = []  # 保持為列表的列表的鍵

    if 'input_text_tks_padded' in first_item and \
       isinstance(first_item['input_text_tks_padded'], dict) and \
       'input_ids' in first_item['input_text_tks_padded']:
        keys_to_stack.append('input_text_tks_padded') # 特殊處理這個嵌套字典
    
    for key in first_item.keys():
        if key != 'input_text_tks_padded' and isinstance(first_item[key], list):
            # 像 known_cuis, hop1_target_cuis 等是 CUI 字符串列表，它們應該保持為列表的列表
            keys_as_list.append(key)
        elif key != 'input_text_tks_padded' and isinstance(first_item[key], str): # 例如 case_id
            keys_as_list.append(key)
        # 其他類型的數據，如果需要特殊處理，可以在這裡添加邏輯

    for key in keys_as_list:
        collated_batch[key] = [item[key] for item in batch]

    if 'input_text_tks_padded' in keys_to_stack:
        input_ids_list = [item['input_text_tks_padded']['input_ids'] for item in batch]
        attention_mask_list = [item['input_text_tks_padded']['attention_mask'] for item in batch]
        
        # 因為 __getitem__ 返回的 input_ids 和 attention_mask 已經是 padding 好的固定長度張量
        # 所以這裡可以直接用 torch.stack
        collated_batch['input_text_tks_padded'] = {
            "input_ids": torch.stack(input_ids_list, dim=0),
            "attention_mask": torch.stack(attention_mask_list, dim=0)
        }
    
    return collated_batch

class FocalLoss(nn.Module):
    """
    Focal Loss 的 PyTorch 實現，用於二元分類任務。
    它旨在解決類別不均衡問題，通過降低對易分類樣本的權重，
    讓模型更專注於難分類的樣本。
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float): 平衡正負樣本權重的因子，範圍在 [0, 1]。
                           常用於處理正負樣本數量不均衡。
            gamma (float): 聚焦參數，用於調節對難易樣本的關注程度。
                           gamma > 0 可以降低對易分類樣本的損失貢獻。
            reduction (str): 指定如何對輸出的損失進行聚合，
                             可選 'mean', 'sum', 'none'。
        """
        super(FocalLoss, self).__init__()
        if not (0 <= alpha <= 1):
            raise ValueError("alpha 參數必須在 0 到 1 之間")
        if not (gamma >= 0):
            raise ValueError("gamma 參數必須大於等於 0")
            
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        計算 Focal Loss。
        Args:
            inputs (torch.Tensor): 模型的原始輸出 (logits)，形狀為 [N, *]。
            targets (torch.Tensor): 真實標籤 (0或1)，形狀與 inputs 相同。
        Returns:
            torch.Tensor: 計算出的損失。
        """
        # 使用 BCEWithLogitsLoss 來獲得更好的數值穩定性
        # reduction='none' 讓我們能對每個元素的損失進行後續操作
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # 獲取模型預測為正類的機率
        p = torch.sigmoid(inputs)

        # 計算 p_t，即模型對真實類別的預測機率
        # 如果 target 是 1，p_t = p
        # 如果 target 是 0，p_t = 1 - p
        p_t = p * targets + (1 - p) * (1 - targets)

        # 計算 focal loss 的核心項：(1 - p_t)^gamma
        focal_term = (1 - p_t).pow(self.gamma)

        # 計算 alpha 項，用於平衡正負樣本
        # 如果 target 是 1，alpha_t = alpha
        # 如果 target 是 0，alpha_t = 1 - alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # 計算最終的 focal loss
        loss = alpha_t * focal_term * bce_loss

        # 根據設定的 reduction 方式進行聚合
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else: # 'none'
            return loss

# ====================== gnn_utils ===================
# Graph utils functions 

def preprocess_graph_to_tensors(graph_nx):
    global mock_cui_to_idx, mock_edge_to_idx
    
    print("Preprocessing graph to tensors...")
    nodes = sorted(list(graph_nx.nodes()))
    mock_cui_to_idx = {cui: i for i, cui in enumerate(nodes)}
    
    edge_labels = sorted(list(set(data['label'] for _, _, data in graph_nx.edges(data=True) if 'label' in data)))
    mock_edge_to_idx = {label: i for i, label in enumerate(edge_labels)}
    if not mock_edge_to_idx and graph_nx.number_of_edges() > 0 : # Handle case with edges but no labels
        mock_edge_to_idx["DEFAULT_REL"] = 0


    num_nodes = len(nodes)
    adj_src = []
    adj_tgt = []
    adj_edge_type = []

    for u, v, data in graph_nx.edges(data=True):
        if u in mock_cui_to_idx and v in mock_cui_to_idx:
            adj_src.append(mock_cui_to_idx[u])
            adj_tgt.append(mock_cui_to_idx[v])
            label = data.get('label')
            if label in mock_edge_to_idx:
                adj_edge_type.append(mock_edge_to_idx[label])
            elif "DEFAULT_REL" in mock_edge_to_idx: # Use default if label missing or unknown
                 adj_edge_type.append(mock_edge_to_idx["DEFAULT_REL"])
            else: # Should not happen if DEFAULT_REL is set up
                adj_edge_type.append(0) # Fallback

    target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not adj_src: # Handle empty graph or graph with no valid mapped edges
        print("Warning: No valid edges found or graph is empty after preprocessing.")
        return {
            "num_nodes": num_nodes,
            "cui_to_idx": mock_cui_to_idx,
            "idx_to_cui": {i: cui for cui, i in mock_cui_to_idx.items()},
            "edge_to_idx": mock_edge_to_idx,
            "idx_to_edge": {i: label for label, i in mock_edge_to_idx.items()},
            "adj_src": torch.tensor([], dtype=torch.long, device=target_device),
            "adj_tgt": torch.tensor([], dtype=torch.long, device=target_device),
            "adj_edge_type": torch.tensor([], dtype=torch.long, device=target_device)
        }

    tensor_graph = {
        "num_nodes": num_nodes,
        "cui_to_idx": mock_cui_to_idx,
        "idx_to_cui": {i: cui for cui, i in mock_cui_to_idx.items()},
        "edge_to_idx": mock_edge_to_idx,
        "idx_to_edge": {i: label for label, i in mock_edge_to_idx.items()},
        "adj_src": torch.tensor(adj_src, dtype=torch.long, device=target_device),
        "adj_tgt": torch.tensor(adj_tgt, dtype=torch.long, device=target_device),
        "adj_edge_type": torch.tensor(adj_edge_type, dtype=torch.long, device=target_device)
    }
    print(f"Preprocessing complete. Num nodes: {num_nodes}, Num edges: {len(adj_src)}")
    return tensor_graph

# --- 重構後的 retrieve_neighbors_paths_no_self_tensorized ---
def retrieve_neighbors_paths_no_self_tensorized(
    current_cui_str_list, 
    tensor_graph, 
    prev_candidate_tensors=None # 這個現在可能包含 "selected_first_hop_edge_idx"
):
    device = tensor_graph['adj_src'].device
    if not current_cui_str_list:
        return torch.tensor([], dtype=torch.long, device=device), \
               torch.tensor([], dtype=torch.long, device=device), \
               torch.tensor([], dtype=torch.long, device=device), \
               None, \
               None # ## ADDED: for mem_first_edge_idx_hop

    cui_to_idx = tensor_graph['cui_to_idx']
    current_cui_indices = [cui_to_idx[s] for s in current_cui_str_list if s in cui_to_idx]
    if not current_cui_indices:
        return torch.tensor([], dtype=torch.long, device=device), \
               torch.tensor([], dtype=torch.long, device=device), \
               torch.tensor([], dtype=torch.long, device=device), \
               None, \
               None # ## ADDED

    current_cui_indices_tensor = torch.tensor(list(set(current_cui_indices)), dtype=torch.long, device=device)

    path_src_list, path_tgt_list, path_edge_type_list = [], [], []
    path_memory_src_prev_hop_list = []
    path_memory_first_edge_idx_list = [] # ## ADDED: 儲存第一跳邊索引的列表

    adj_src_graph = tensor_graph['adj_src']
    adj_tgt_graph = tensor_graph['adj_tgt']
    adj_edge_type_graph = tensor_graph['adj_edge_type']

    for i, src_idx_current_hop in enumerate(current_cui_indices_tensor):
        mask_src_is_current = (adj_src_graph == src_idx_current_hop)
        if not torch.any(mask_src_is_current):
            continue

        srcs_for_paths = adj_src_graph[mask_src_is_current]
        tgts_for_paths = adj_tgt_graph[mask_src_is_current]
        edge_types_for_paths = adj_edge_type_graph[mask_src_is_current]
        
        num_new_paths = srcs_for_paths.size(0)

        path_src_list.append(srcs_for_paths)
        path_tgt_list.append(tgts_for_paths)
        path_edge_type_list.append(edge_types_for_paths)

        current_hop_memory_source = src_idx_current_hop # 默認為當前節點 (用於第一跳或回退)
        current_hop_first_edge_memory = None # ## ADDED: 默認為 None

        if prev_candidate_tensors is not None and \
           'selected_src_orig_idx' in prev_candidate_tensors and \
           prev_candidate_tensors['selected_src_orig_idx'] is not None and \
           'selected_hop_target_idx' in prev_candidate_tensors:
            
            mask_in_prev_targets = (prev_candidate_tensors['selected_hop_target_idx'] == src_idx_current_hop)
            if torch.any(mask_in_prev_targets):
                original_source_candidates = prev_candidate_tensors['selected_src_orig_idx'][mask_in_prev_targets]
                if original_source_candidates.numel() > 0:
                    current_hop_memory_source = original_source_candidates[0]
                
                # ## ADDED: 檢查並獲取第一跳的邊索引
                if 'selected_first_hop_edge_idx' in prev_candidate_tensors and \
                   prev_candidate_tensors['selected_first_hop_edge_idx'] is not None:
                    first_edge_candidates = prev_candidate_tensors['selected_first_hop_edge_idx'][mask_in_prev_targets]
                    if first_edge_candidates.numel() > 0:
                        current_hop_first_edge_memory = first_edge_candidates[0]
        
        path_memory_src_prev_hop_list.append(current_hop_memory_source.repeat(num_new_paths))
        if current_hop_first_edge_memory is not None: # ## ADDED
            path_memory_first_edge_idx_list.append(current_hop_first_edge_memory.repeat(num_new_paths))
        else: # ## ADDED: 如果沒有第一跳邊信息，用一個特殊值填充，例如 -1，或保持與路徑數量匹配的 None 列表
              # 這裡用 -1 填充，假設邊索引都是非負的
            path_memory_first_edge_idx_list.append(torch.full((num_new_paths,), -1, dtype=torch.long, device=device))


    if not path_src_list:
        return torch.tensor([], dtype=torch.long, device=device), \
               torch.tensor([], dtype=torch.long, device=device), \
               torch.tensor([], dtype=torch.long, device=device), \
               None, \
               None # ## ADDED

    candidate_src_indices = torch.cat(path_src_list)
    candidate_tgt_indices = torch.cat(path_tgt_list)
    candidate_edge_type_indices = torch.cat(path_edge_type_list)
    
    path_memory_src_prev_hop_tensor = torch.cat(path_memory_src_prev_hop_list) if path_memory_src_prev_hop_list else None
    path_memory_first_edge_idx_tensor = torch.cat(path_memory_first_edge_idx_list) if path_memory_first_edge_idx_list else None # ## ADDED
    
    non_self_loop_mask = (candidate_src_indices != candidate_tgt_indices)
    candidate_src_indices = candidate_src_indices[non_self_loop_mask]
    candidate_tgt_indices = candidate_tgt_indices[non_self_loop_mask]
    candidate_edge_type_indices = candidate_edge_type_indices[non_self_loop_mask]

    if path_memory_src_prev_hop_tensor is not None:
        if path_memory_src_prev_hop_tensor.size(0) == non_self_loop_mask.size(0):
            path_memory_src_prev_hop_tensor = path_memory_src_prev_hop_tensor[non_self_loop_mask]
    
    if path_memory_first_edge_idx_tensor is not None: # ## ADDED
        if path_memory_first_edge_idx_tensor.size(0) == non_self_loop_mask.size(0):
            path_memory_first_edge_idx_tensor = path_memory_first_edge_idx_tensor[non_self_loop_mask]

    return candidate_src_indices.to(device), \
           candidate_tgt_indices.to(device), \
           candidate_edge_type_indices.to(device), \
           path_memory_src_prev_hop_tensor.to(device) if path_memory_src_prev_hop_tensor is not None else None, \
           path_memory_first_edge_idx_tensor.to(device) if path_memory_first_edge_idx_tensor is not None else None # ## ADDED

# ====================== gnn  ===================

class CuiEmbedding(object):
    """
    Loads pre-computed CUI embeddings from a pickle file.
    Handles values that are numpy.ndarray of shape (1, embedding_dim).
    """
    def __init__(self, embedding_file_path, device=torch.device('cpu')):
        print(f"載入 CUI 嵌入從: {embedding_file_path}")
        try:
            with open(embedding_file_path, 'rb') as f:
                self.raw_data = pickle.load(f) # CUI_str -> numpy.ndarray (1,768)
            print(f"成功載入 {len(self.raw_data)} 個 CUI 嵌入。")
        except FileNotFoundError:
            print(f"錯誤：找不到 CUI 嵌入文件 {embedding_file_path}。")
            raise
        except Exception as e:
            print(f"載入 CUI 嵌入文件時發生錯誤: {e}")
            raise
        
        self.device = device
        self.data = {} # 將存儲 CUI_str -> torch.Tensor [embedding_dim]
        self.embedding_dim = None

        if not self.raw_data:
            print("警告: 載入的 CUI 嵌入數據為空。")
            return

        # Convert loaded embeddings to tensors on the target device
        # and reshape from (1, 768) to (768)
        for cui, emb_array in self.raw_data.items():
            if not isinstance(cui, str):
                print(f"警告: 發現非字串类型的 CUI key '{cui}' ({type(cui)})，已跳過。")
                continue

            if isinstance(emb_array, np.ndarray):
                if emb_array.shape == (1, 768): # 檢查形狀是否如您所述
                    if self.embedding_dim is None:
                        self.embedding_dim = emb_array.shape[1]
                    elif self.embedding_dim != emb_array.shape[1]:
                        print(f"警告: CUI '{cui}' 的嵌入維度 ({emb_array.shape[1]}) 與之前 ({self.embedding_dim}) 不符，已跳過。")
                        continue
                    
                    # 轉換為 Tensor, squeeze, 並移動到設備
                    self.data[cui] = torch.from_numpy(emb_array).float().squeeze(0).to(self.device)
                else:
                    print(f"警告: CUI '{cui}' 的 NumPy 嵌入形狀 ({emb_array.shape}) 不是預期的 (1, 768)，已跳過。")
                    continue
            elif isinstance(emb_array, torch.Tensor): # 如果 pkl 中直接存的是 Tensor
                if emb_array.shape == (1, 768) or emb_array.shape == (768,):
                    if self.embedding_dim is None:
                        self.embedding_dim = emb_array.shape[-1]
                    elif self.embedding_dim != emb_array.shape[-1]:
                        print(f"警告: CUI '{cui}' 的 Tensor 嵌入維度 ({emb_array.shape[-1]}) 與之前 ({self.embedding_dim}) 不符，已跳過。")
                        continue
                    
                    self.data[cui] = emb_array.float().reshape(-1).to(self.device) # Reshape to [768]
                else:
                    print(f"警告: CUI '{cui}' 的 Tensor 嵌入形狀 ({emb_array.shape}) 不是預期的 (1, 768) 或 (768,)，已跳過。")
                    continue
            else:
                print(f"警告: CUI '{cui}' 的嵌入格式未知 ({type(emb_array)})，已跳過。")
                continue
        
        if not self.data:
            print("警告: 沒有成功轉換任何 CUI 嵌入。請檢查 pkl 檔案內容和格式。")
        elif self.embedding_dim is None and self.data: # Should have been set if data exists
             # Fallback: try to infer from first item if not set but data exists (should not happen with current logic)
             self.embedding_dim = next(iter(self.data.values())).shape[0]


    def encode(self, cui_str_list: list):
        """
        Encodes a list of CUI strings into their embedding tensors.
        Returns a tensor of shape [len(cui_str_list_found), embedding_dim].
        Handles missing CUIs by returning zero vectors.
        """
        embeddings_list = []
        if self.embedding_dim is None: # 如果没有任何有效嵌入被加载
            print("錯誤: CuiEmbedding 未能確定嵌入維度。無法編碼。")
            # 返回一个明确表示错误的形状，或者根据期望的hdim创建一个空的
            # 假设下游期望一个hdim，即使这里无法确定，也应该返回一个有意义的空tensor
            # 但更安全的是，如果embedding_dim未定义，表明初始化失败。
            # 理想情况下，如果__init__后self.data为空，self.embedding_dim也应为None
            # 这里的hdim_fallback仅用于避免在encode中直接崩溃，但表示了更深层的问题
            hdim_fallback = 768 # 或者从一个配置中获取
            print(f"警告: 在 encode 中回退到 hdim={hdim_fallback} 因为 self.embedding_dim 未设置。")
            for cui_str in cui_str_list: # 即使无法查找，也为每个请求的CUI生成占位符
                 embeddings_list.append(torch.zeros(hdim_fallback, device=self.device))
            if not embeddings_list:
                 return torch.empty(0, hdim_fallback, device=self.device)
            return torch.stack(embeddings_list)


        for cui_str in cui_str_list:
            emb = self.data.get(cui_str)
            if emb is not None:
                # 確保 emb 是一維的 [embedding_dim]
                if emb.dim() == 1 and emb.shape[0] == self.embedding_dim:
                    embeddings_list.append(emb)
                else:
                    print(f"警告: CUI '{cui_str}' 的內部存儲嵌入形狀異常 ({emb.shape})，期望 ({self.embedding_dim},)。使用零向量。")
                    embeddings_list.append(torch.zeros(self.embedding_dim, device=self.device))
            else:
                # print(f"警告: CUI '{cui_str}' 在嵌入字典中未找到。使用零向量代替。")
                embeddings_list.append(torch.zeros(self.embedding_dim, device=self.device))

        if not embeddings_list:
            return torch.empty(0, self.embedding_dim, device=self.device)
        
        return torch.stack(embeddings_list) # 返回 [N, embedding_dim]

    def to(self, device):
        self.device = device
        # 重新處理 self.raw_data 以確保所有 Tensor 都移動到新設備
        new_data_on_device = {}
        if hasattr(self, 'raw_data') and self.raw_data: # 檢查 raw_data 是否存在且非空
            for cui, emb_array in self.raw_data.items():
                if not isinstance(cui, str): continue

                if isinstance(emb_array, np.ndarray) and emb_array.shape == (1, self.embedding_dim if self.embedding_dim else 768):
                    new_data_on_device[cui] = torch.from_numpy(emb_array).float().squeeze(0).to(self.device)
                elif isinstance(emb_array, torch.Tensor) and (emb_array.shape == (1, self.embedding_dim if self.embedding_dim else 768) or emb_array.shape == (self.embedding_dim if self.embedding_dim else 768,)):
                    new_data_on_device[cui] = emb_array.float().reshape(-1).to(self.device)
                # else: # 跳過格式不符的
            self.data = new_data_on_device
        else: # 如果 raw_data 為空或不存在，則嘗試移動 self.data 中的現有 Tensor (如果有的話)
            current_data_temp = self.data.copy() # 複製以避免在迭代時修改
            self.data = {}
            for cui, emb_tensor in current_data_temp.items():
                 self.data[cui] = emb_tensor.to(self.device)

        return self
    
class EdgeEmbeddingLookup(object):
    """
    從 pickle 文件加載預先計算好的關係嵌入，並將其構建成一個全局嵌入矩陣，
    以便進行高效的、基於索引的張量查找。
    """
    def __init__(self, embedding_file_path, edge_to_idx, device=torch.device('cpu')):
        print(f"Loading and building relation embedding matrix from: {embedding_file_path}")
        
        # 加載原始的 "名稱->嵌入" 字典
        try:
            with open(embedding_file_path, 'rb') as f:
                raw_data = pickle.load(f)
        except Exception as e:
            print(f"ERROR: Failed to load relation embedding file at {embedding_file_path}. Error: {e}")
            raise
            
        self.device = device
        self.embedding_dim = next(iter(raw_data.values())).shape[-1] if raw_data else 768

        # 創建一個空的全局嵌入矩陣
        num_relations = len(edge_to_idx)
        self.global_relation_embedding_matrix = torch.zeros(num_relations, self.embedding_dim, device=self.device)

        # 根據 edge_to_idx 的順序，填充這個矩陣
        for name, index in edge_to_idx.items():
            embedding_vector = raw_data.get(name)
            if embedding_vector is not None:
                self.global_relation_embedding_matrix[index] = torch.from_numpy(embedding_vector).float().to(self.device)
            else:
                print(f"Warning: Relation '{name}' found in graph but not in embedding file. Using zero vector.")
        
        print(f"Global relation embedding matrix created with shape: {self.global_relation_embedding_matrix.shape}")

    def lookup_by_index(self, edge_indices_tensor: torch.Tensor) -> torch.Tensor:
        """
        根據整數索引張量，直接從全局矩陣中查找嵌入。這是一個高效的 GPU 操作。
        """
        # 直接使用 PyTorch 的索引功能，這會在 GPU 上並行完成
        return self.global_relation_embedding_matrix[edge_indices_tensor]
    
    

    def to(self, device):
        self.device = device
        for name, emb in self.data.items():
            self.data[name] = emb.to(device)
        return self


class EdgeOneHot(object): # Using the dynamic one from user's code
    def __init__(self, graph: nx.DiGraph, device, unknown_rel_label: str = "UNKNOWN_REL"):
        super().__init__()
        unique_edge_labels = set()
        for _, _, data in graph.edges(data=True):
            label = data.get('label')
            unique_edge_labels.add(label if label is not None else unknown_rel_label)
        if not unique_edge_labels and graph.number_of_edges() > 0: # If edges exist but no labels
            unique_edge_labels.add(unknown_rel_label)

        self.edge_mappings = {label: i for i, label in enumerate(sorted(list(unique_edge_labels)))}
        self.num_edge_types = len(self.edge_mappings)
        self.unknown_rel_index = self.edge_mappings.get(unknown_rel_label)
        self.device = device
        
        if self.num_edge_types == 0: # Handle graph with no edges or no labeled edges
            self.onehot_mat = torch.empty(0,0, device=self.device).float() # Or torch.zeros(1,1).float() if a min dim is needed
        else:
            self.onehot_mat = F.one_hot(torch.arange(0, self.num_edge_types, device=self.device), num_classes=self.num_edge_types).float()
    def Lookup(self, edge_indices_tensor: torch.Tensor):
        # device = self.onehot_mat.device # 確保返回張量在正確設備
        if self.num_edge_types == 0: # 處理沒有邊類型的情況
            # 如果下游期望一個特定的維度（例如 actual_edge_dim），即使是0，也要匹配
            # 假設若 num_edge_types 為0，則 actual_edge_dim 也應為0
            # （或者在 __init__ 中將 num_edge_types 設為至少1，並有一個 UNKNOWN 類型）
            return torch.empty(edge_indices_tensor.size(0), 0, device=self.device) 

        # 確保索引在有效範圍內，防止 OOB 錯誤
        clamped_indices = torch.clamp(edge_indices_tensor, 0, self.num_edge_types - 1)
        
        return self.onehot_mat[clamped_indices]

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp_layer1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.mlp_layer2 = nn.Linear(hidden_dim, output_dim, bias=False)
        nn.init.xavier_uniform_(self.mlp_layer1.weight)
        nn.init.xavier_uniform_(self.mlp_layer2.weight)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        h = x
        if h.shape[0] == 1:
            h = F.relu(self.mlp_layer1(h))
        else:
            h = F.relu(self.batch_norm(self.mlp_layer1(h)))
        return self.mlp_layer2(h)
    
class GINStack(nn.Module):
    def __init__(self, input_node_dim, input_edge_dim, hidden_dim, output_dim_final, num_layers, device, learn_eps=False):
        super().__init__()
        self.convs = nn.ModuleList()
        current_dim = input_node_dim
        for _ in range(num_layers):
            self.convs.append(NodeAggregateGIN(current_dim, input_edge_dim, hidden_dim, hidden_dim, device, learn_eps=learn_eps))
            current_dim = hidden_dim # Output of GIN layer becomes input for next

        # Linear layers to combine outputs from different GIN layers (if needed, or just use last layer's output)
        # Original DR.KNOWS concatenated outputs of 3 layers. Here, we'll use a simpler sequential stack.
        self.lin_out = nn.Linear(hidden_dim * num_layers, output_dim_final) # Example: concat all layer outputs
        # Or, if just using the last layer's output:
        # self.lin_out = nn.Linear(hidden_dim, output_dim_final)
        self.num_layers = num_layers

    def forward(self, x_src_unique, unique_src_to_process_indices,
                path_source_indices_global_scatter, # This is the index for scatter_add
                path_target_node_features,
                path_edge_features):
        
        layer_outputs = []
        h = x_src_unique # Initial features for GIN layers

        for i in range(self.num_layers):
            # Pass the current node features `h` and the neighborhood info for these nodes
            h, _ = self.convs[i](
                h, # current features of the unique source nodes
                unique_src_to_process_indices, # global indices of these nodes
                path_source_indices_global_scatter, # scatter index for paths
                path_target_node_features,
                path_edge_features
            )
            layer_outputs.append(h)
        
        # Concatenate outputs from all layers (like original DR.KNOWS)
        h_concat = torch.cat(layer_outputs, dim=1)
        final_output = self.lin_out(h_concat)
        # Or, if only using the last layer's output:
        # final_output = self.lin_out(h)
        
        return final_output, unique_src_to_process_indices

class NodeAggregateGIN(nn.Module):
    def __init__(self, input_node_dim, input_edge_dim, hidden_dim, output_dim, device, init_eps=0, learn_eps=False):
        super().__init__()
        # Linear layer for transforming combined (neighbor_node_feat + edge_feat)
        self.edge_transform_linear = nn.Linear(input_node_dim + input_edge_dim, hidden_dim)
        # MLP for aggregation
        self.mlp = MLP(hidden_dim, hidden_dim, output_dim) # MLP aggregates sum_of_transformed_neighbors
        self.device = device
        if learn_eps:
            self.eps = nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([init_eps]))
        self.output_dim = output_dim


    def forward(self, x_src_unique, # [num_unique_src, src_feat_dim] - Features of source nodes to be updated
                unique_src_to_process_indices, # [num_unique_src] - Global indices of these source nodes
                # Information about paths/edges originating from *these* unique_src_nodes
                path_source_indices_global,    # [total_paths_from_these_srcs] - Global index of the source for each path
                path_target_node_features, # [total_paths_from_these_srcs, tgt_feat_dim] - Features of target nodes of these paths
                path_edge_features         # [total_paths_from_these_srcs, edge_feat_dim] - Features of edges of these paths
               ):
        # `x_src_unique` are features of the nodes we want to update.
        # `path_source_indices_global` tells us which source each path belongs to.
        # We need to map `path_source_indices_global` to local indices relative to `x_src_unique`.

        if not TORCH_SCATTER_AVAILABLE:
            # Fallback for environments without torch_scatter
            # This will be slow and is a simplified placeholder for the logic
            # print("GIN Fallback: Using looped aggregation (inefficient).")
            aggregated_msgs = torch.zeros(x_src_unique.size(0), self.edge_transform_linear.out_features,
                                          device=self.device, dtype=x_src_unique.dtype)
            
            # Create a mapping from global source index to its local index in x_src_unique
            map_global_src_idx_to_local = {
                global_idx.item(): local_idx for local_idx, global_idx in enumerate(unique_src_to_process_indices)
            }

            for i in range(path_source_indices_global.size(0)):
                src_global_idx = path_source_indices_global[i].item()
                if src_global_idx in map_global_src_idx_to_local:
                    src_local_idx = map_global_src_idx_to_local[src_global_idx]
                    
                    # Combine neighbor (target) and edge features for this path
                    combined_neighbor_edge_feat = torch.cat(
                        (path_target_node_features[i], path_edge_features[i]), dim=-1
                    ).unsqueeze(0) # Add batch dim for linear layer
                    
                    transformed_msg = F.relu(self.edge_transform_linear(combined_neighbor_edge_feat))
                    aggregated_msgs[src_local_idx] += transformed_msg.squeeze(0)
        else:
            
            combined_neighbor_edge_feats = torch.cat((path_target_node_features, path_edge_features), dim=-1)
                     
            transformed_messages = F.relu(self.edge_transform_linear(combined_neighbor_edge_feats))
            
            if path_source_indices_global.max() >= x_src_unique.size(0):
                 print(f"Error: scatter index max {path_source_indices_global.max()} out of bounds for dim_size {x_src_unique.size(0)}")
                 # Handle error or return
                 # For now, fallback to un-updated features or zeros
                 updated_src_features = self.mlp(x_src_unique) # Or just x_src_unique
                 return updated_src_features, unique_src_to_process_indices # Return global indices


            aggregated_msgs = scatter_add(
                transformed_messages,         # Embeddings of all neighbors+edges
                path_source_indices_global,   # Index mapping each neighbor+edge to its source node (0 to N-1)
                dim=0,                        # Aggregate along dimension 0
                dim_size=x_src_unique.size(0) # Size of the output tensor (num unique source nodes)
            ) # Shape: [num_unique_src, hidden_dim]
            
        # GIN update rule
        # print(f"x_src_unique device: {x_src_unique.device}, aggregated_msgs device: {aggregated_msgs.device}, eps device: {self.eps.device}")
        updated_src_features = (1 + self.eps) * x_src_unique + aggregated_msgs
        updated_src_features = self.mlp(updated_src_features)

        return updated_src_features, unique_src_to_process_indices # Return updated features and their global indices


class GATLayer(nn.Module):
    """
    能夠整合邊特徵的圖注意力網路層 (Relational GAT 的一種簡化實現)。
    """
    def __init__(self, node_in_features, edge_in_features, out_features, n_heads, concat=True, dropout=0.1, leaky_relu_negative_slope=0.2):
        super(GATLayer, self).__init__()
        self.n_heads = n_heads
        self.out_features = out_features
        self.concat = concat

        # 線性變換層
        self.W_node = nn.Linear(node_in_features, n_heads * out_features, bias=False)
        self.W_edge = nn.Linear(edge_in_features, n_heads * out_features, bias=False)
        
        # 注意力機制參數
        self.a = nn.Parameter(torch.Tensor(n_heads, 2 * out_features))

        self.leaky_relu = nn.LeakyReLU(leaky_relu_negative_slope)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)

        nn.init.xavier_uniform_(self.W_node.weight, gain=1.414)
        nn.init.xavier_uniform_(self.W_edge.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h_node, edge_index, h_edge):
        """
        Args:
            h_node (Tensor): 節點特徵矩陣 [num_nodes, node_in_features]
            edge_index (Tensor): 邊索引 [2, num_edges]
            h_edge (Tensor): 邊特徵矩陣 [num_edges, edge_in_features]
        """
        if not TORCH_SCATTER_AVAILABLE:
            raise ImportError("EdgeGATLayer requires torch_scatter for efficient operation.")

        # 1. 對節點和邊特徵進行線性變換
        h_node_transformed = self.W_node(h_node).view(-1, self.n_heads, self.out_features)
        h_edge_transformed = self.W_edge(h_edge).view(-1, self.n_heads, self.out_features)

        source_nodes_features = h_node_transformed[edge_index[0]] # [num_edges, n_heads, out_features]
        target_nodes_features = h_node_transformed[edge_index[1]] # [num_edges, n_heads, out_features]

        # 2. 計算注意力分數（核心修改：將邊的資訊加入）
        # 將源節點特徵與邊特徵相加，作為資訊傳遞的「內容」
        message_content = source_nodes_features + h_edge_transformed
        attn_input = torch.cat([message_content, target_nodes_features], dim=-1) # [num_edges, n_heads, 2 * out_features]
        e = self.leaky_relu((attn_input * self.a).sum(dim=-1)) # [num_edges, n_heads]

        # 3. 使用 scatter_softmax 進行歸一化
        attention = scatter_softmax(e, edge_index[1], dim=0)
        attention = self.dropout(attention)

        # 4. 聚合鄰居節點特徵
        # 將注意力權重應用到我們構造的 message_content 上
        h_prime_scatter = message_content * attention.unsqueeze(-1)

        # 使用 scatter_add 聚合加權後的特徵到目標節點
        h_prime = scatter_add(h_prime_scatter, edge_index[1], dim=0, dim_size=h_node.size(0))

        if self.concat:
            return F.elu(h_prime.view(-1, self.n_heads * self.out_features))
        else:
            return F.elu(h_prime.mean(dim=1))

### 【修改點 2C】：創建對應的 EdgeGATStack
class GATStack(nn.Module):
    """多層 EdgeGAT 的堆疊。"""
    def __init__(self, node_in_features, edge_in_features, hidden_features, out_features, num_layers, n_heads):
        super(GATStack, self).__init__()
        self.layers = nn.ModuleList()
        # 輸入層
        self.layers.append(GATLayer(node_in_features, edge_in_features, hidden_features, n_heads=n_heads, concat=True))
        # 隱藏層
        for _ in range(num_layers - 2):
            self.layers.append(GATLayer(hidden_features * n_heads, edge_in_features, hidden_features, n_heads=n_heads, concat=True))
        # 輸出層
        self.layers.append(GATLayer(hidden_features * n_heads, edge_in_features, out_features, n_heads=1, concat=False))

    def forward(self, h_node, edge_index, h_edge):
        for layer in self.layers:
            h_node = layer(h_node, edge_index, h_edge)
        # 返回更新後的節點嵌入和一個佔位符 None 以匹配 GINStack 的輸出簽名
        return h_node, None



class PathEncoder(nn.Module):
    """
    Generate path embedding given src node emb and (target + edge) embedding 
    module has been tested 
    """
    def __init__(self, hdim, path_dim):
        super(PathEncoder, self).__init__()
        self.d = hdim 
        self.src_weights = nn.Linear(hdim, hdim)
        self.tgt_weights = nn.Linear(path_dim, hdim)
        self.batch_norm = nn.BatchNorm1d((hdim))

        nn.init.xavier_uniform_(self.src_weights.weight)
        nn.init.xavier_uniform_(self.tgt_weights.weight)

    def forward(self, src, tgt):
        #print("SRC weight update"torch.sum)
        hpath = self.src_weights(src) + self.tgt_weights(tgt)
        if hpath.shape[0] == 1:
            hpath = F.relu(hpath, inplace=True)
        else:
            hpath = F.relu(self.batch_norm(hpath), inplace=True)
        return hpath # B X D

class PathEncoderTransformer(nn.Module):
    """
    Generate path embedding given src node emb and (target + edge) embedding 
    module has been tested 
    """
    def __init__(self, hdim, path_dim):
        super().__init__()
        self.d = hdim 
        #self.src_weights = nn.Linear(hdim, hdim)
        self.tgt_transform = nn.Linear(path_dim, hdim) # input is target+edge, output is hdim 
        nn.init.xavier_uniform_(self.tgt_transform.weight)

        self.path_encoder = nn.Transformer(d_model=hdim,
                                           nhead=3,
                                           num_encoder_layers=1,
                                           num_decoder_layers=1,
                                                                                   dim_feedforward=128,
                                          batch_first=True) 

    def forward(self, src, tgt):
        # input src: a list of source nodes 
        htgt = self.tgt_transform(tgt) # output is B x 768 paths, where B is batch size 
        htgt = htgt.view(htgt.shape[0], 1, htgt.shape[-1]) # reshape to B X 1 X 768
        #print("HTGT shape", htgt.shape)
        #print("SRC SHAPE", src.shape) # expected B X L X 768
        hpath = self.path_encoder(src, htgt) 

        return hpath # B X D


class TriAttnFlatPathRanker(nn.Module):
    """
    Input: task embedding, cui embedding, and path embedding 
    Trilinear Attention module for path ranker  
    Flatten trilinear attention 
    """
    def __init__(self, hdim):
        super(TriAttnFlatPathRanker, self).__init__()

        self.w1 = nn.Parameter(torch.Tensor(3*hdim, 3*hdim))
        self.w2 = nn.Parameter(torch.Tensor(3*hdim, 3*hdim))
        self.w3 = nn.Parameter(torch.Tensor(3*hdim, hdim))
        self.out = nn.Parameter(torch.Tensor(hdim, 1))

        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.xavier_uniform_(self.w3)
        nn.init.xavier_uniform_(self.out)

    def forward(self, h_text, h_con, h_path):
        x = torch.cat([h_text, h_con, h_path], dim=-1)
        x = torch.matmul(x, self.w1)
        x = torch.matmul(x, self.w2)
        x = F.relu(torch.matmul(x, self.w3))
        out = torch.matmul(x, self.out) 

        return out 


class TriAttnCombPathRanker(nn.Module):
    """
    Input: task embedding, cui embedding, and path embedding 
    Trilinear Attention module for path ranker  
    Weighted combination of trilinear attention 
    """
    def __init__(self, hdim):
        super(TriAttnCombPathRanker, self).__init__()

        self.w1 = nn.Parameter(torch.Tensor(hdim, hdim))
        self.w2 = nn.Parameter(torch.Tensor(hdim, hdim))
        self.w3 = nn.Parameter(torch.Tensor(hdim, hdim))
        self.out = nn.Parameter(torch.Tensor(hdim, 1))

        bias = torch.empty(1)
        torch.nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.xavier_uniform_(self.w3)
        nn.init.xavier_uniform_(self.out)

    def forward(self, h_text, h_con, h_path):
        
        w_text = torch.matmul(h_text, self.w1)
        w_con = torch.matmul(h_con, self.w2)
        w_path = torch.matmul(h_path, self.w3)
        res = w_text + w_con + w_path 
        res += self.bias 
        out = torch.matmul(res, self.out) 

        return out 



# ====================== model  ===================

class GraphModel(nn.Module):
    def __init__(self,
                 g_nx, # NetworkX graph for preprocessing
                 cui_embedding_lookup, # CuiEmbedding object (lookup by CUI string)
                 relation_embedding_filepath, 
                 hdim,
                 nums_of_head,
                 num_hops, # Max hops
                 top_n,
                 device,
                 cui_weights_dict=None, # Dict: CUI_str -> weight
                 gnn_update=True,
                 path_encoder_type="MLP",
                 path_ranker_type="Flat",
                 gnn_type="Stack",
                 gin_hidden_dim=None,
                 gin_num_layers=1,
                 #input_edge_dim_for_gin=108
                 ):
        super(GraphModel, self).__init__()
        self.g_tensorized = preprocess_graph_to_tensors(g_nx) # Preprocess NX graph
        self.n_encoder_lookup = cui_embedding_lookup # For string CUI to embedding
        self.device = device
        
        
        num_graph_nodes = self.g_tensorized['num_nodes']
        idx_to_cui_map_for_init = self.g_tensorized['idx_to_cui']

        all_embeddings_list = []
        found_count = 0
        if self.n_encoder_lookup.embedding_dim is None and num_graph_nodes > 0:
            # 嘗試從第一個有效CUI推斷維度，如果CuiEmbedding的dim未設置
            # 這是一個回退，理想情況下CuiEmbedding.__init__應能確定dim
            first_valid_cui_for_dim_check = idx_to_cui_map_for_init.get(0)
            if first_valid_cui_for_dim_check:
                temp_emb_for_dim_check = self.n_encoder_lookup.encode([first_valid_cui_for_dim_check])
                if temp_emb_for_dim_check.numel() > 0:
                    self.hdim = temp_emb_for_dim_check.shape[-1] # 更新hdim以匹配實際嵌入維度
                    print(f"GraphModel: hdim 已根據實際嵌入更新為 {self.hdim}")


        print(f"GraphModel: 正在為 {num_graph_nodes} 個節點創建全局嵌入矩陣...")
        for i in range(num_graph_nodes):
            cui_str = idx_to_cui_map_for_init.get(i)
            if cui_str is not None:
                # CuiEmbedding.encode 返回 [1, hdim] (如果輸入是列表) 或 [hdim]
                # 我們需要確保是一致的 [hdim]
                emb_tensor = self.n_encoder_lookup.encode([cui_str]) # 輸入列表以獲得批次輸出
                if emb_tensor.numel() > 0:
                    all_embeddings_list.append(emb_tensor.squeeze(0)) # 移除批次維度
                    found_count += 1
                else: # CUI存在於圖中但不在嵌入字典中
                    print(f"警告: CUI '{cui_str}' (索引 {i}) 在圖中但未找到其嵌入，將使用零向量。")
                    all_embeddings_list.append(torch.zeros(self.hdim, device=self.device))
            else: # 理論上不應發生，因為 idx_to_cui_map 應包含所有 0 到 num_nodes-1 的索引
                print(f"警告: 全局CUI索引 {i} 未在 idx_to_cui_map 中找到，將使用零向量。")
                all_embeddings_list.append(torch.zeros(self.hdim, device=self.device))

        if all_embeddings_list:
            self.global_cui_embedding_matrix = torch.stack(all_embeddings_list).to(self.device)
            print(f"GraphModel: 全局嵌入矩陣創建完畢，形狀: {self.global_cui_embedding_matrix.shape}，找到 {found_count}/{num_graph_nodes} 個嵌入。")
        else: # 如果圖中沒有節點，或者所有節點都沒有嵌入
            print(f"GraphModel: 未能創建全局嵌入矩陣 (可能是圖中無節點或無嵌入)。")
            self.global_cui_embedding_matrix = torch.empty(0, self.hdim, device=self.device)

        # 將 global_cui_embedding_matrix 註冊為緩衝區，使其隨模型移動到不同設備，且不被視為可訓練參數
        self.register_buffer('global_cui_embedding_matrix_buffer', self.global_cui_embedding_matrix)
        # 在使用時，可以用 self.global_cui_embedding_matrix_buffer 替代 self.global_cui_embedding_matrix
        # 或者在 __init__ 結束時 self.global_cui_embedding_matrix = self.global_cui_embedding_matrix_buffer
                
  
        # self.e_encoder = EdgeOneHot(graph=g_nx, device=self.device) # Assumes EdgeOneHot can handle NX graph
        self.e_encoder = EdgeEmbeddingLookup(
            embedding_file_path=relation_embedding_filepath, # 或者從參數傳入
            edge_to_idx=self.g_tensorized['edge_to_idx'],
            device=self.device
        )
        actual_edge_dim = self.e_encoder.embedding_dim if hasattr(self.e_encoder, 'embedding_dim') and self.e_encoder.embedding_dim is not None else hdim
        
        self.p_encoder_type = path_encoder_type
        self.path_ranker_type = path_ranker_type
        
        
        self.edge_to_node_projection_for_transformer = None
        if self.p_encoder_type == "Transformer":
            self.edge_to_node_projection_for_transformer = nn.Linear(actual_edge_dim, hdim)
            nn.init.xavier_uniform_(self.edge_to_node_projection_for_transformer.weight)
            if self.edge_to_node_projection_for_transformer.bias is not None:
                 nn.init.zeros_(self.edge_to_node_projection_for_transformer.bias)
            self.p_encoder = PathEncoderTransformer(hdim, hdim + actual_edge_dim) # tgt 輸入維度不變
        else: # MLP
            self.p_encoder = PathEncoder(hdim, hdim + actual_edge_dim)

        if self.path_ranker_type == "Combo":
            self.p_ranker = TriAttnCombPathRanker(hdim)
        else:
            self.p_ranker = TriAttnFlatPathRanker(hdim)

        self.k_hops = num_hops
        self.path_per_batch_size = 64
        self.top_n = top_n
        self.cui_weights_dict = cui_weights_dict if cui_weights_dict else {}
        self.hdim = hdim # Store hdim for use
        
        self.gnn_update = gnn_update
        self.gnn_type = gnn_type
        self.gin_num_layers = gin_num_layers if gin_num_layers else (2 if gnn_type == "Stack" else 1)
        self.gin_hidden_dim = gin_hidden_dim if gin_hidden_dim else hdim
        #self.input_edge_dim_for_gin = input_edge_dim_for_gin # Should match e_encoder output
        self.gnn_update = gnn_update
        if self.gnn_update:
            if gnn_type.upper() == "GAT":
                print("Using GAT as the GNN backend.")
                # 假設 GAT 的輸入維度是 hdim
                # n_heads 可以作為一個新的超參數
                self.gnn = GATStack(
                    node_in_features=hdim, 
                    edge_in_features=actual_edge_dim, # 使用 SapBERT 邊嵌入維度
                    hidden_features=gin_hidden_dim // nums_of_head,
                    out_features=hdim,
                    num_layers=gin_num_layers,
                    n_heads=nums_of_head
                )
            elif gnn_type.upper() == "STACK": # GINStack
                print("Using GINStack as the GNN backend.")
                self.gnn = GINStack(
                    input_node_dim=hdim, 
                    input_edge_dim=actual_edge_dim, 
                    hidden_dim=self.gin_hidden_dim, 
                    output_dim_final=hdim, 
                    num_layers=self.gin_num_layers, 
                    device=device
                )
            else: # 默認或單層GIN
                print("Using single layer NodeAggregateGIN as the GNN backend.")
                self.gnn = NodeAggregateGIN(
                    input_node_dim=hdim, 
                    input_edge_dim=actual_edge_dim, 
                    hidden_dim=self.gin_hidden_dim, 
                    output_dim=hdim, 
                    device=device
                )
        else:
            print("GNN update is disabled.")
            self.gnn = None
            

        

    def _get_embeddings_by_indices(self, cui_indices_tensor):
        """
        直接從預存的全局嵌入矩陣中通過索引獲取嵌入。
        Args:
            cui_indices_tensor (torch.Tensor): 包含全局CUI索引的1D Tensor。
        Returns:
            torch.Tensor: 形狀為 [len(cui_indices_tensor), hdim] 的嵌入張量。
        """
        if cui_indices_tensor is None or cui_indices_tensor.numel() == 0:
            return torch.empty(0, self.hdim, device=self.device) # 返回正確形狀的空張量

        # 確保索引在有效範圍內 (可選，但推薦)
        # clamped_indices = torch.clamp(cui_indices_tensor, 0, self.global_cui_embedding_matrix.size(0) - 1)
        # return self.global_cui_embedding_matrix[clamped_indices]

        # 或者，假設傳入的 cui_indices_tensor 總是有效的
        return self.global_cui_embedding_matrix[cui_indices_tensor]


    def one_iteration(self,
                      task_emb_batch, # Shape [1, hdim] or [hdim] for a single sample
                      current_cui_str_list, # List of CUI strings for current hop's start nodes
                      running_k_hop, # Current hop number (0 for 1st hop)
                      context_emb_batch=None, # Shape [1, hdim] or [hdim]
                      prev_iteration_state=None, # Dict: {'cand_src_orig_idx': Tensor, 'cand_tgt_idx': Tensor}
                      gt_indices_for_pruning=None
                     ):
        stop_flag = False
        
        
        # 1. Retrieve 1-hop paths using tensorized function
        # prev_candidate_tensors in retrieve_... is prev_iteration_state
        cand_src_idx_hop, cand_tgt_idx_hop, cand_edge_idx_hop, \
        mem_orig_src_idx_hop, mem_first_edge_idx_hop = \
            retrieve_neighbors_paths_no_self_tensorized(
                current_cui_str_list,
                self.g_tensorized,
                prev_iteration_state
            )
        pre_pruning_cand_tgt_idx_hop = cand_tgt_idx_hop.clone() if cand_tgt_idx_hop is not None else torch.empty(0, dtype=torch.long, device=self.device)
        debug_info = {'pre_pruning_targets': pre_pruning_cand_tgt_idx_hop}
        if cand_src_idx_hop.numel() == 0: # No paths found
            return None, {}, None, True, debug_info # Scores, next_hop_dict, path_tensors, mem_tensors, stop_flag
        
        num_paths_this_hop = cand_src_idx_hop.size(0)

        # --- 2. Prepare Embeddings for this Hop ---
        # Unique source and target CUI indices for this hop's paths
        unique_hop_src_indices = torch.unique(cand_src_idx_hop)
        unique_hop_tgt_indices = torch.unique(cand_tgt_idx_hop)
        unique_mem_orig_src_indices = torch.unique(mem_orig_src_idx_hop) if mem_orig_src_idx_hop is not None else None

        # Get embeddings for these unique CUIs
        # These are the base embeddings before GIN (if any)
        unique_src_embs_base = self._get_embeddings_by_indices(unique_hop_src_indices)
        unique_tgt_embs = self._get_embeddings_by_indices(unique_hop_tgt_indices)
        unique_mem_orig_src_embs = self._get_embeddings_by_indices(unique_mem_orig_src_indices) if unique_mem_orig_src_indices is not None else None

        if unique_src_embs_base.numel() == 0 or unique_tgt_embs.numel() == 0:
            print(f"Debug: Failed to get base embeddings for src or tgt at hop {running_k_hop}")
            return None, {}, None, True, debug_info
        
        # 獲取圖譜中最大的CUI索引值，以確定映射張量的大小
        # 假設 self.g_tensorized['num_nodes'] 是節點總數，索引從 0 到 num_nodes-1
        max_cui_idx_in_graph = self.g_tensorized['num_nodes'] - 1 

        # --- 針對 unique_hop_src_indices 的映射 ---
        # 初始化映射張量，填充-1（表示該全局索引不在當前的unique集合中）
        map_global_src_to_local_tensor = torch.full((max_cui_idx_in_graph + 1,), -1, dtype=torch.long, device=self.device)
        if unique_hop_src_indices.numel() > 0: # 僅當 unique_hop_src_indices 非空時操作
            local_indices_for_src = torch.arange(unique_hop_src_indices.numel(), device=self.device)
            map_global_src_to_local_tensor[unique_hop_src_indices] = local_indices_for_src

        # --- 針對 unique_hop_tgt_indices 的映射 ---
        map_global_tgt_to_local_tensor = torch.full((max_cui_idx_in_graph + 1,), -1, dtype=torch.long, device=self.device)
        if unique_hop_tgt_indices.numel() > 0: # 僅當 unique_hop_tgt_indices 非空時操作
            local_indices_for_tgt = torch.arange(unique_hop_tgt_indices.numel(), device=self.device)
            map_global_tgt_to_local_tensor[unique_hop_tgt_indices] = local_indices_for_tgt

        # --- 針對 unique_mem_orig_src_indices 的映射 (如果 Transformer 會用到) ---
        map_global_mem_orig_src_to_local_tensor = torch.full((max_cui_idx_in_graph + 1,), -1, dtype=torch.long, device=self.device)
        if unique_mem_orig_src_indices is not None and unique_mem_orig_src_indices.numel() > 0:
            local_indices_for_mem_orig_src = torch.arange(unique_mem_orig_src_indices.numel(), device=self.device)
            map_global_mem_orig_src_to_local_tensor[unique_mem_orig_src_indices] = local_indices_for_mem_orig_src
            
        # map_global_src_idx_to_local_in_unique = {glob_idx.item(): i for i, glob_idx in enumerate(unique_hop_src_indices)}
        # This scatter_index maps each path's global source CUI to its 0..N-1 index within unique_hop_src_indices
        # scatter_src_index_for_gin_agg = map_global_src_to_local_tensor[cand_src_idx_hop]

        # Get edge embeddings for all paths
        # EdgeOneHot.Lookup expects list of labels, convert cand_edge_idx_hop back to labels
        # --- 在 cand_edge_idx_hop 被定義之後 ---
        # 現在 EdgeOneHot.Lookup 直接接收索引張量
        # cand_edge_idx_hop 本身就是邊類型的索引 (0 到 num_edge_types-1)
        if cand_edge_idx_hop.numel() > 0:
            
            path_edge_embs_for_gin_and_path_enc = self.e_encoder.lookup_by_index(cand_edge_idx_hop)
        else: 
            print("error no edge exists")
        
        
        # --- 3. Optional GNN Update on current_path_src_embs ---
        # This part is complex to tensorize fully without specific GIN assumptions.
        # We need to aggregate target+edge features for each source in unique_hop_src_indices.
        if self.gnn_update and self.gnn:
            
            # --- 【修改】根據 GNN 類型準備不同的輸入並調用 ---
            if self.gnn_type.upper() == "GAT":
                # GAT 需要一個包含所有參與計算節點的特徵矩陣，以及一個描述連接關係的 edge_index。
                
                # 1. 準備 GAT 的節點特徵輸入 (h)
                # 合併當前跳所有涉及的源節點和目標節點，並去重
                all_involved_unique_indices, inverse_map = torch.unique(
                    torch.cat([unique_hop_src_indices, unique_hop_tgt_indices]), return_inverse=True
                )
                # 獲取這些唯一節點的嵌入
                all_involved_embs = self._get_embeddings_by_indices(all_involved_unique_indices)

                # 2. 準備 GAT 的邊索引輸入 (edge_index)
                # 創建從全局CUI索引到GAT局部索引（0 到 N-1）的映射
                map_global_to_gat_local = torch.full((self.g_tensorized['num_nodes'],), -1, dtype=torch.long, device=self.device)
                map_global_to_gat_local[all_involved_unique_indices] = torch.arange(all_involved_unique_indices.size(0), device=self.device)

                # 使用映射將全局邊索引轉換為局部邊索引
                gat_edge_index_src = map_global_to_gat_local[cand_src_idx_hop]
                gat_edge_index_tgt = map_global_to_gat_local[cand_tgt_idx_hop]
                
                # 過濾掉無效的邊（如果某個節點因為某些原因沒有被包含在 all_involved_unique_indices 中）
                valid_edge_mask = (gat_edge_index_src != -1) & (gat_edge_index_tgt != -1)
                gat_edge_index = torch.stack([
                    gat_edge_index_src[valid_edge_mask], 
                    gat_edge_index_tgt[valid_edge_mask]
                ], dim=0)
                
                gat_edge_features = path_edge_embs_for_gin_and_path_enc[valid_edge_mask]
                
                # 3. 調用 GAT
                if all_involved_embs.numel() > 0 and gat_edge_index.numel() > 0:
                    # GAT 會更新所有參與節點的嵌入
                    updated_all_involved_embs, _ = self.gnn(all_involved_embs, gat_edge_index, gat_edge_features)
                    
                    # 4. 從更新後的嵌入中，取出我們需要的源節點嵌入
                    local_indices_for_final_srcs = map_global_to_gat_local[unique_hop_src_indices]
                    
                    # 再次檢查，確保所有源節點都有有效的局部索引
                    if torch.any(local_indices_for_final_srcs == -1):
                        raise ValueError("GAT 更新後，部分源節點無法找到其對應的局部索引。")

                    current_path_src_embs_for_encoding = updated_all_involved_embs[local_indices_for_final_srcs]
                else:
                    # 如果沒有有效的節點或邊來運行GAT，則不更新
                    print(f"警告 (GAT): 在 hop {running_k_hop}，沒有有效的節點或邊來運行GAT，將跳過GNN更新。")

            else: # GIN 的邏輯
                # GIN 需要的是源節點特徵，以及一個將所有路徑信息聚合回源節點的 scatter_index。
                
                # 1. 準備 GIN 的 scatter_index
                scatter_src_index_for_gin_agg = map_global_src_to_local_tensor[cand_src_idx_hop]
                if torch.any(scatter_src_index_for_gin_agg == -1):
                    raise ValueError("GIN 輸入準備錯誤：cand_src_idx_hop 中存在無法映射到局部索引的節點。")

                # 2. 準備 GIN 的鄰居特徵 (目標節點)
                local_indices_for_gin_targets = map_global_tgt_to_local_tensor[cand_tgt_idx_hop]
                if torch.any(local_indices_for_gin_targets == -1):
                    raise ValueError("GIN 輸入準備錯誤：cand_tgt_idx_hop 中存在無法映射到局部索引的節點。")
                
                gin_path_tgt_node_features = torch.empty(0, self.hdim, device=self.device)
                if unique_tgt_embs.numel() > 0:
                    gin_path_tgt_node_features = unique_tgt_embs[local_indices_for_gin_targets]

                # 3. 確保所有輸入的維度一致
                if gin_path_tgt_node_features.size(0) == path_edge_embs_for_gin_and_path_enc.size(0) and \
                   scatter_src_index_for_gin_agg.size(0) == gin_path_tgt_node_features.size(0):
                    
                    # 4. 調用 GIN
                    updated_src_embs_from_gin, _ = self.gnn(
                        current_path_src_embs_for_encoding, 
                        unique_hop_src_indices,            
                        scatter_src_index_for_gin_agg,     
                        gin_path_tgt_node_features,        
                        path_edge_embs_for_gin_and_path_enc
                    )
                    current_path_src_embs_for_encoding = updated_src_embs_from_gin
                else:
                    # 如果輸入維度不匹配，跳過GNN更新以防出錯
                    print(f"警告 (GIN): 在 hop {running_k_hop}，輸入維度不匹配，將跳過GNN更新。")
            
        pruning_threshold_count = 64 # 您設定的篩選路徑數量上限
        num_paths_this_hop_before_pruning = num_paths_this_hop # ## NOW THIS IS VALID ##

        if num_paths_this_hop > pruning_threshold_count: # Check against initial num_paths_this_hop
            # print(f"  Hop {running_k_hop}: Path count {num_paths_this_hop} exceeds threshold {pruning_threshold_count}. Applying pruning.")

            if unique_tgt_embs.numel() > 0 : # 確保 unique_tgt_embs 不是空的
                if gt_indices_for_pruning is not None and gt_indices_for_pruning.numel() > 0:
                    
                    
                    # 1. 識別並分離出 GT 路徑
                    is_gt_mask = torch.isin(cand_tgt_idx_hop, gt_indices_for_pruning)
                    gt_path_indices = torch.where(is_gt_mask)[0]
                    non_gt_path_indices = torch.where(~is_gt_mask)[0]
                    
                    # 2. 確定需要從非 GT 路徑中保留多少個
                    num_to_keep_from_non_gt = pruning_threshold_count - len(gt_path_indices)
                    
                    if num_to_keep_from_non_gt > 0 and len(non_gt_path_indices) > 0:
                        # 3. 只對非GT路徑計算相似度分數並選取Top-K
                        non_gt_cand_tgt_idx = cand_tgt_idx_hop[non_gt_path_indices]
                        
                        # 獲取非GT目標的嵌入
                        # (這部分需要從之前的 unique_tgt_embs 中索引，此處為簡化邏輯)
                        # 假設 map_global_tgt_to_local_tensor 和 unique_tgt_embs 已準備好
                        local_indices = map_global_tgt_to_local_tensor[non_gt_cand_tgt_idx]
                        valid_mask = (local_indices != -1)
                        path_specific_tgt_embs_for_pruning = unique_tgt_embs[local_indices[valid_mask]]
                        
                        expanded_task_emb = task_emb_batch.expand(path_specific_tgt_embs_for_pruning.size(0), -1)
                        non_gt_similarity_scores = F.cosine_similarity(path_specific_tgt_embs_for_pruning, expanded_task_emb, dim=1)
                        
                        num_to_sample = min(num_to_keep_from_non_gt, len(non_gt_similarity_scores))
                        _, top_k_relative_indices = torch.topk(non_gt_similarity_scores, num_to_sample)
                        
                        # 獲取在原始非GT索引列表中的絕對索引
                        top_k_non_gt_indices_absolute = non_gt_path_indices[valid_mask][top_k_relative_indices]
                        
                        # 4. 合併GT路徑和被選中的非GT路徑的索引
                        final_indices_to_keep = torch.cat([gt_path_indices, top_k_non_gt_indices_absolute])
                    else:
                        # 如果GT路徑就已經超過閾值，或沒有非GT路徑，則只保留GT路徑
                        final_indices_to_keep = gt_path_indices
                    
                    # 5. 使用最終索引來篩選所有相關張量
                    cand_src_idx_hop = cand_src_idx_hop[final_indices_to_keep]
                    cand_tgt_idx_hop = cand_tgt_idx_hop[final_indices_to_keep]
                    cand_edge_idx_hop = cand_edge_idx_hop[final_indices_to_keep]
                    if mem_orig_src_idx_hop is not None: mem_orig_src_idx_hop = mem_orig_src_idx_hop[final_indices_to_keep]
                    if mem_first_edge_idx_hop is not None: mem_first_edge_idx_hop = mem_first_edge_idx_hop[final_indices_to_keep]
                    path_edge_embs_for_gin_and_path_enc = path_edge_embs_for_gin_and_path_enc[final_indices_to_keep]
                    num_paths_this_hop = cand_src_idx_hop.size(0)
                else:    
                
                    local_indices_for_pruning_targets = map_global_tgt_to_local_tensor[cand_tgt_idx_hop] # 直接用映射張量

                    # 處理可能的 -1 索引 (如果 cand_tgt_idx_hop 中有不在 unique_hop_tgt_indices 的索引)
                    valid_mask_for_pruning_targets = (local_indices_for_pruning_targets != -1)
                    if not torch.all(valid_mask_for_pruning_targets):
                        print(f"警告 (Pruning Input): cand_tgt_idx_hop 中存在無效索引，將被過濾或導致錯誤。")
                        # 應對策略：只對有效索引的路徑進行後續操作，或者如果這是嚴重錯誤則拋出異常
                        # 這裡假設如果出現-1，對應的相似度會很低或被忽略，或者 topk 會處理
                        # 為了安全，最好確保所有索引都有效，或者過濾掉帶-1索引的路徑
                        # local_indices_for_pruning_targets = local_indices_for_pruning_targets[valid_mask_for_pruning_targets]
                        # expanded_task_emb 和 path_specific_tgt_embs_for_pruning 也需要對應篩選
                        # 這會使邏輯複雜。更簡單的是假設 cand_tgt_idx_hop 中的索引總能在 map_global_tgt_to_local_tensor 中找到有效映射。
                        if torch.any(local_indices_for_pruning_targets == -1):
                            raise ValueError("Pruning Input: 發現無效的-1索引，cand_tgt_idx_hop 與 unique_hop_tgt_indices 不一致。")

                    path_specific_tgt_embs_for_pruning = unique_tgt_embs[local_indices_for_pruning_targets]
                
                    expanded_task_emb_for_pruning = task_emb_batch.expand(path_specific_tgt_embs_for_pruning.size(0), -1)
                    target_similarity_scores = F.cosine_similarity(path_specific_tgt_embs_for_pruning, expanded_task_emb_for_pruning, dim=1)
                    
                    # Ensure pruning_threshold_count isn't larger than available paths
                    actual_pruning_count = min(pruning_threshold_count, target_similarity_scores.size(0))
                    if actual_pruning_count > 0:
                        _, top_k_pruning_indices = torch.topk(target_similarity_scores, actual_pruning_count)

                        cand_src_idx_hop = cand_src_idx_hop[top_k_pruning_indices]
                        cand_tgt_idx_hop = cand_tgt_idx_hop[top_k_pruning_indices]
                        cand_edge_idx_hop = cand_edge_idx_hop[top_k_pruning_indices]
                        if mem_orig_src_idx_hop is not None:
                            mem_orig_src_idx_hop = mem_orig_src_idx_hop[top_k_pruning_indices]
                        if mem_first_edge_idx_hop is not None:
                            mem_first_edge_idx_hop = mem_first_edge_idx_hop[top_k_pruning_indices]
                        
                        path_edge_embs_for_gin_and_path_enc = path_edge_embs_for_gin_and_path_enc[top_k_pruning_indices]
                        
                        num_paths_this_hop = cand_src_idx_hop.size(0) # ## UPDATE num_paths_this_hop AFTER PRUNING ##
                        # print(f"  Hop {running_k_hop}: Pruned from {num_paths_this_hop_before_pruning} to {num_paths_this_hop} paths.")
                    else: # No paths left after trying to select top_k (e.g. target_similarity_scores was empty)
                        # print(f"  Hop {running_k_hop}: Pruning resulted in 0 paths to keep.")
                        # Set all path tensors to empty
                        cand_src_idx_hop = torch.empty(0, dtype=torch.long, device=self.device)
                        # ... set other cand_... and mem_... tensors to empty as well ...
                        num_paths_this_hop = 0

            else: # unique_tgt_embs was empty, cannot perform similarity pruning
                print(f"  Hop {running_k_hop}: Skipping pruning because unique_tgt_embs is empty.")
            

        # --- 5. Path Encoding and Ranking (Mini-batch loop) ---
        # num_paths_this_hop 現在是剪枝後的數量（或原始數量，如果未達到剪枝閾值）
        # all_path_scores_hop, all_encoded_paths_hop = [], [] # 移到迴圈前

        # 準備 PathEncoder 的輸入 (src_b, combined_tgt_edge_b)
        # 這些應基於剪枝後的 cand_src_idx_hop, cand_tgt_idx_hop, path_edge_embs_for_gin_and_path_enc
        
        # --- 在剪枝邏輯之後，mini-batch 迴圈之前 ---
        # map_global_src_to_local_tensor 是基於剪枝前的 unique_hop_src_indices 創建的
        # current_path_src_embs_for_encoding 的行序與剪枝前的 unique_hop_src_indices 對應
        # cand_src_idx_hop 此時是剪枝後的路徑的源節點全局索引

        local_indices_for_path_src = map_global_src_to_local_tensor[cand_src_idx_hop] # cand_src_idx_hop 是剪枝後的

        if torch.any(local_indices_for_path_src == -1):
            print(f"警告 (PathEncoder Src Input): 剪枝後的 cand_src_idx_hop 中存在未映射到 unique_hop_src_indices 的索引。")
            # 應對策略：
            if torch.all(local_indices_for_path_src == -1): # 如果所有都無效
                 path_specific_src_embs = torch.empty(0, self.hdim, device=self.device) # 產生空張量
            else: # 過濾掉無效的，只保留有效的
                valid_mask = (local_indices_for_path_src != -1)
                # 如果過濾，則所有與路徑對應的張量（cand_tgt_idx_hop, cand_edge_idx_hop 等）都需要同步篩選
                # 這會使剪枝後的第二次篩選邏輯複雜化。
                # 更簡單的處理（但可能引入錯誤數據）：將-1映射到一個有效索引（如0）或用0向量填充
                # 假設：剪枝後的 cand_src_idx_hop 中的所有索引都應該能在 map_global_src_to_local_tensor 中找到有效映射
                raise ValueError("PathEncoder Src Input: 發現無效的-1索引，這在剪枝後不應發生。")
        
        path_specific_src_embs = current_path_src_embs_for_encoding[local_indices_for_path_src]
        
        # --- 在 path_specific_src_embs 準備之後 ---
        # map_global_tgt_to_local_tensor 是基於剪枝前的 unique_hop_tgt_indices 創建的
        # unique_tgt_embs 的行序與剪枝前的 unique_hop_tgt_indices 對應
        # cand_tgt_idx_hop 此時是剪枝後的路徑的目標節點全局索引

        local_indices_for_path_tgt_encoding = map_global_tgt_to_local_tensor[cand_tgt_idx_hop] # cand_tgt_idx_hop 是剪枝後的

        if torch.any(local_indices_for_path_tgt_encoding == -1):
            print(f"警告 (PathEncoder Tgt Input): 剪枝後的 cand_tgt_idx_hop 中存在未映射到 unique_hop_tgt_indices 的索引。")
            # 類似地，需要錯誤處理策略
            raise ValueError("PathEncoder Tgt Input: 發現無效的-1索引，這在剪枝後不應發生。")

        # 確保 unique_tgt_embs 非空，並且索引有效
        if unique_tgt_embs.numel() > 0 :
            path_specific_tgt_embs_for_encoding = unique_tgt_embs[local_indices_for_path_tgt_encoding]
        elif cand_tgt_idx_hop.numel() > 0 : # 有剪枝後的目標索引，但 unique_tgt_embs 最初就為空
             print(f"警告: unique_tgt_embs 為空，無法為PathEncoder獲取有效的 path_specific_tgt_embs_for_encoding。")
             path_specific_tgt_embs_for_encoding = torch.empty(0, self.hdim, device=self.device)
        else: # 剪枝後 cand_tgt_idx_hop 也為空
            path_specific_tgt_embs_for_encoding = torch.empty(0, self.hdim, device=self.device)


        # path_edge_embs_for_gin_and_path_enc 應該已經在剪枝步驟 4d 中被同步篩選過了
        # 確保它的行數與 path_specific_tgt_embs_for_encoding 一致
        if path_specific_tgt_embs_for_encoding.size(0) == path_edge_embs_for_gin_and_path_enc.size(0) and \
           path_specific_tgt_embs_for_encoding.numel() > 0 : # 確保不對空張量進行cat
            combined_tgt_edge_embs_for_path_enc = torch.cat(
                (path_specific_tgt_embs_for_encoding, path_edge_embs_for_gin_and_path_enc), dim=-1
            )
        elif cand_tgt_idx_hop.numel() > 0 : # 如果剪枝後仍有路徑，但嵌入準備有問題
            print(f"警告: PathEncoder 的目標嵌入或邊嵌入準備不一致或為空。 Tgt shape: {path_specific_tgt_embs_for_encoding.shape}, Edge shape: {path_edge_embs_for_gin_and_path_enc.shape}")
            # 創建一個正確形狀的空張量或零張量，以避免後續 mini-batch 迴圈出錯
            # actual_edge_dim 需要從 self.e_encoder.num_edge_types 獲取
            _actual_edge_dim = self.e_encoder.num_edge_types if self.e_encoder.num_edge_types > 0 else 0
            combined_tgt_edge_embs_for_path_enc = torch.empty(0, self.hdim + _actual_edge_dim, device=self.device)
            if path_specific_tgt_embs_for_encoding.size(0) != path_edge_embs_for_gin_and_path_enc.size(0) and path_specific_tgt_embs_for_encoding.numel() > 0 and path_edge_embs_for_gin_and_path_enc.numel() > 0:
                 raise ValueError("PathEncoder Tgt/Edge Input: 維度不匹配，即使在嘗試處理後。")

        else: # 剪枝後沒有路徑了
            _actual_edge_dim = self.e_encoder.num_edge_types if self.e_encoder.num_edge_types > 0 else 0
            combined_tgt_edge_embs_for_path_enc = torch.empty(0, self.hdim + _actual_edge_dim, device=self.device)

        # mini-batch 迴圈現在處理的是剪枝後的路徑
        all_path_scores_hop, all_encoded_paths_hop = [], [] # 重置
        for i in range(0, num_paths_this_hop, self.path_per_batch_size):
            s_ = slice(i, min(i + self.path_per_batch_size, num_paths_this_hop))
            
            # --- Transformer 輸入序列構建 (如果 path_encoder_type == "Transformer") ---
            # 這部分邏輯需要使用剪枝後的索引和嵌入
            # mem_orig_src_idx_hop, mem_first_edge_idx_hop, cand_src_idx_hop 都已被剪枝
            # unique_mem_orig_src_embs, current_path_src_embs_for_encoding (源於 unique_src_embs_base) 仍是基於剪枝前的 unique 索引
            # 所以在構建序列時，需要小心地使用這些映射
            
            if self.p_encoder_type == "Transformer":
                # --- 1. 為當前 mini-batch 批量準備所有需要的嵌入 ---
                # 這些 cand_* 和 mem_* 張量都已經是剪枝後的了，s_ 是對它們的切片

                # a. 最初始源節點的全局索引 (對於當前 mini-batch)
                batch_mem_orig_src_idx = mem_orig_src_idx_hop[s_] if mem_orig_src_idx_hop is not None else None
                
                # b. 中間節點 (即當前跳的源節點) 的全局索引 (對於當前 mini-batch)
                batch_curr_src_idx = cand_src_idx_hop[s_] 
                
                # c. 第ㄧ跳邊的全局類型索引 (對於當前 mini-batch)
                batch_mem_first_edge_idx = mem_first_edge_idx_hop[s_] if mem_first_edge_idx_hop is not None else None

                # 獲取這些索引對應的嵌入
                # unique_mem_orig_src_embs 是基於剪枝前的 unique_mem_orig_src_indices
                # map_global_mem_orig_src_to_local_tensor 也是基於剪枝前的
                batch_orig_src_embs = None
                if batch_mem_orig_src_idx is not None and unique_mem_orig_src_embs is not None and unique_mem_orig_src_embs.numel() > 0:
                    local_indices = map_global_mem_orig_src_to_local_tensor[batch_mem_orig_src_idx]
                    valid_mask = (local_indices != -1)
                    if torch.any(valid_mask):
                        batch_orig_src_embs = torch.zeros(batch_mem_orig_src_idx.size(0), self.hdim, device=self.device)
                        batch_orig_src_embs[valid_mask] = unique_mem_orig_src_embs[local_indices[valid_mask]]
                
                # current_path_src_embs_for_encoding 是基於剪枝前的 unique_hop_src_indices (GNN可能更新過)
                # map_global_src_to_local_tensor 也是基於剪枝前的
                local_indices_curr = map_global_src_to_local_tensor[batch_curr_src_idx]
                if torch.any(local_indices_curr == -1): raise ValueError("Transformer Input: 無效的batch_curr_src_idx映射")
                batch_curr_src_embs = current_path_src_embs_for_encoding[local_indices_curr]

                batch_projected_first_rel_embs = None
                if running_k_hop == 1 and batch_mem_first_edge_idx is not None and self.edge_to_node_projection_for_transformer is not None:
                    valid_first_edge_mask = (batch_mem_first_edge_idx != -1) 
                    if torch.any(valid_first_edge_mask):
                        batch_projected_first_rel_embs = torch.zeros(batch_mem_first_edge_idx.size(0), self.hdim, device=self.device)
                       
                        first_rel_indices_to_lookup = batch_mem_first_edge_idx[valid_first_edge_mask]
                        
                        if first_rel_indices_to_lookup.numel() > 0:
                            first_rel_sapbert_embs = self.e_encoder.lookup_by_index(first_rel_indices_to_lookup)                   
                            batch_projected_first_rel_embs[valid_first_edge_mask] = self.edge_to_node_projection_for_transformer(first_rel_sapbert_embs)
                
                # --- 2. Python 迴圈逐條構建序列 (內部操作現在是從預提取的批次嵌入中索引) ---
                src_sequences_for_transformer = []
                for j_in_batch in range(batch_curr_src_idx.size(0)): # 遍歷 mini-batch 中的路徑
                    path_elements_embs = []
                    
                    # 獲取當前路徑的最初始源節點的全局索引 (仍然需要 .item() 進行條件判斷)
                    orig_src_idx_val_current_path = batch_mem_orig_src_idx[j_in_batch].item() if batch_mem_orig_src_idx is not None else -1
                    curr_src_idx_val_current_path = batch_curr_src_idx[j_in_batch].item()


                    # 1. 添加最初始源節點嵌入 (CUI_A)
                    added_orig_src = False
                    if batch_orig_src_embs is not None: # batch_orig_src_embs 已經是 mini-batch 對應的嵌入了
                        path_elements_embs.append(batch_orig_src_embs[j_in_batch])
                        added_orig_src = True
                    else: # 如果是第一跳，batch_mem_orig_src_idx 可能與 batch_curr_src_idx 相同（或其記憶為自身）
                        # 或者 batch_orig_src_embs 因為某些原因沒有成功獲取
                        path_elements_embs.append(batch_curr_src_embs[j_in_batch]) # 此時 curr_src 就是最初始源
                        added_orig_src = True # 視為已添加最初始源（即當前源）

                    # 2. 添加第一個關係的嵌入 (Rel1) - 僅當是第二跳且 Rel1 有效時
                    if running_k_hop == 1:
                        if batch_projected_first_rel_embs is not None:
                            # batch_mem_first_edge_idx[j_in_batch].item() 仍然需要判斷是否為 -1
                            # 以確定 batch_projected_first_rel_embs[j_in_batch] 是否為有效投影嵌入或零向量
                            if batch_mem_first_edge_idx[j_in_batch].item() != -1:
                                path_elements_embs.append(batch_projected_first_rel_embs[j_in_batch])
                            else: # 無效的 Rel1，添加零向量占位符
                                path_elements_embs.append(torch.zeros(self.hdim, device=self.device))
                        else: # 如果 batch_projected_first_rel_embs 本身就是 None
                            path_elements_embs.append(torch.zeros(self.hdim, device=self.device))


                    # 3. 添加中間節點嵌入 (CUI_B) - 僅當是第二跳且與最初始源節點不同時
                    if running_k_hop == 1:
                        # 僅當 orig_src 和 curr_src 不同時，curr_src (中間節點) 才作為序列的獨立元素添加
                        # 如果是第一跳，curr_src 已經在上面作為 "最初始源" 添加了
                        # 如果 orig_src 因為某些原因沒有被添加到 path_elements_embs (added_orig_src is False)
                        # 或者 orig_src 和 curr_src 不同，則需要添加 curr_src
                        if not added_orig_src or \
                        (orig_src_idx_val_current_path != -1 and \
                            orig_src_idx_val_current_path != curr_src_idx_val_current_path):
                            path_elements_embs.append(batch_curr_src_embs[j_in_batch])
                    
                    if path_elements_embs:
                        src_sequences_for_transformer.append(torch.stack(path_elements_embs))
                    else: # 理論上不應發生，因為至少有一個源節點嵌入
                        src_sequences_for_transformer.append(torch.zeros((1, self.hdim), device=self.device))
                
                # --- 3. 填充序列並準備 src_b ---
                if not src_sequences_for_transformer:
                    src_b = torch.empty(0,0,self.hdim, device=self.device) 
                else:
                    src_b = pad_sequence(src_sequences_for_transformer, batch_first=True, padding_value=0.0)
            else: # MLP Encoder Logic
                # path_specific_src_embs 已經在剪枝後、迴圈前準備好了
                src_b = path_specific_src_embs[s_]
            
            # combined_tgt_edge_embs_for_path_enc 也已經在剪枝後、迴圈前準備好了
            combined_tgt_edge_b = combined_tgt_edge_embs_for_path_enc[s_]

            # --- 後續的編碼與排序邏輯 ---
            # 準備 combined_tgt_edge_b (這部分需要小心，確保索引 s_ 的應用正確)
            # map_global_tgt_idx_to_local_in_unique = {glob_idx.item(): i for i, glob_idx in enumerate(unique_hop_tgt_indices)}
            
            # cand_tgt_idx_hop[s_] 獲取當前 mini-batch 對應的目標節點的全局索引
            current_batch_tgt_global_indices = cand_tgt_idx_hop[s_]
            # 將這些全局索引轉換為在 unique_tgt_embs 中的局部索引
            current_batch_tgt_local_indices = map_global_tgt_to_local_tensor[current_batch_tgt_global_indices]
            path_specific_tgt_embs_for_batch = unique_tgt_embs[current_batch_tgt_local_indices]
            
            combined_tgt_edge_b = torch.cat(
                (path_specific_tgt_embs_for_batch, path_edge_embs_for_gin_and_path_enc[s_]), dim=-1
            )

            if src_b.numel() == 0 or src_b.shape[0] == 0:
                continue

            encoded_b = self.p_encoder(src_b, combined_tgt_edge_b) # src_b現在是序列, tgt是組合
            if self.p_encoder_type == "Transformer" and encoded_b.dim() == 3:
                 # Transformer 的輸出通常是 (batch, seq_len, feature_dim)
                 # 我們需要決定如何從序列輸出中得到單個路徑嵌入
                 # 例如，取序列的第一個token ([CLS]) 的輸出，或對所有token輸出取平均
                 # 假設 PathEncoderTransformer 的輸出已經處理好了，是 [batch_size_path_encoder, hdim]
                 # 如果 PathEncoderTransformer.forward 返回的是 (batch, seq_len, hdim)
                 # 而下游 p_ranker 期望 (batch, hdim)，則需要調整
                 # 原始 DR.KNOWS 的 nn.Transformer(batch_first=True) 用於 encoder-decoder 時，
                 # encoder輸出(memory)是(N,S,E), decoder輸出是(N,T,E)
                 # 這裡 self.p_encoder(src, htgt), htgt 的 seq_len 是 1.
                 # 所以輸出 hpath 的 seq_len 也可能是 1, squeeze(1) 是合理的
                if encoded_b.shape[1] == 1: # 如果序列長度為1
                    encoded_b = encoded_b.squeeze(1)
                else: # 如果序列長度不為1, 例如TransformerEncoder的輸出
                      # 我們可能取第一個元素的表示 (類似CLS)
                    encoded_b = encoded_b[:, 0, :] 


            all_encoded_paths_hop.append(encoded_b)

            task_exp_b = task_emb_batch.expand(encoded_b.size(0), -1)
            ctx_exp_b = context_emb_batch.expand(encoded_b.size(0), -1) if context_emb_batch is not None else task_exp_b
            scores_b = self.p_ranker(task_exp_b, ctx_exp_b, encoded_b)
            all_path_scores_hop.append(scores_b)

        if not all_path_scores_hop:
             return None, {}, None, True

        final_scores_hop = torch.cat(all_path_scores_hop, dim=0)
        encoded_paths_tensor_hop = torch.cat(all_encoded_paths_hop, dim=0)

        top_n_val = min(self.top_n, final_scores_hop.size(0))
        if top_n_val == 0: return None, {}, None, True

        _, top_k_indices = torch.topk(final_scores_hop.squeeze(-1), top_n_val, dim=0)

        sel_tgt_idx = cand_tgt_idx_hop[top_k_indices]
        sel_mem_orig_src_idx = mem_orig_src_idx_hop[top_k_indices] if mem_orig_src_idx_hop is not None else None
        sel_edge_idx_thishop = cand_edge_idx_hop[top_k_indices] # 當前這一跳選出的邊 (Rel2 for 2-hop)

        all_paths_info = {
            "scores": final_scores_hop, "encoded_embeddings": encoded_paths_tensor_hop,
            "src_idx": cand_src_idx_hop, "tgt_idx": cand_tgt_idx_hop,
            "edge_idx": cand_edge_idx_hop, 
            "mem_orig_src_idx": mem_orig_src_idx_hop,
            "mem_first_edge_idx": mem_first_edge_idx_hop
        }
        
        next_hop_state = {
            "selected_src_orig_idx": sel_mem_orig_src_idx,
            "selected_hop_target_idx": sel_tgt_idx,
        }
        if running_k_hop == 0:
            next_hop_state["selected_first_hop_edge_idx"] = sel_edge_idx_thishop
        elif prev_iteration_state and "selected_first_hop_edge_idx" in prev_iteration_state:
            if mem_first_edge_idx_hop is not None:
                # 確保 mem_first_edge_idx_hop 的長度與 top_k_indices 的數量一致
                # 這在 retrieve_neighbors 中已經被 non_self_loop_mask 篩選過了
                if mem_first_edge_idx_hop.size(0) == cand_src_idx_hop.size(0): # cand_src_idx_hop 是篩選前的
                    next_hop_state["selected_first_hop_edge_idx"] = mem_first_edge_idx_hop[top_k_indices]
                else: # 如果長度不一致，這是一個潛在的問題
                    print(f"Warning: Length mismatch for mem_first_edge_idx_hop in next_hop_state. Size: {mem_first_edge_idx_hop.size(0)}, TopK indices num: {top_k_indices.numel()}")
                    # 作為回退，可以不設置，或者設置為 None/特殊值
                    next_hop_state["selected_first_hop_edge_idx"] = None 
            else:
                 next_hop_state["selected_first_hop_edge_idx"] = None


        visited_paths_str_dict = {}
        return all_paths_info, visited_paths_str_dict, next_hop_state, stop_flag, debug_info


class Trainer(nn.Module):
    def __init__(self, tokenizer,
                 encoder, 
                 g_nx, 
                 cui_embedding_lookup,
                 relation_embedding_filepath,
                 hdim,
                 nums_of_head, 
                 cui_vocab_str_to_idx, 
                 top_n, # top_n 仍然用於 GraphModel 內部決定下一跳的探索寬度
                 device,
                 nums_of_epochs,
                 LR,
                 loss_type='BCE', # 新增：可選 'BCE' 或 'Focal'
                 focal_alpha=0.25, # 新增：Focal Loss 的 alpha
                 focal_gamma=2.0,  # 新增：Focal Loss 的 gamma
                 cui_weights_dict=None,
                 contrastive_learning=True,
                 save_model_path=None,
                 gnn_update=True,
                 intermediate=False, 
                 score_threshold=0.5, # ## ADDED: 評分閾值
                 finetune_encoder=True,          # 是否訓練 encoder
                 use_soft_labels=True,           # 是否使用軟標籤
                 soft_label_threshold=0.5,     # 軟標籤的閾值
                 preserve_gt_in_pruning=False,  # 是否在剪枝時保留GT
                 distance_metric="Cosine",
                 path_encoder_type="Transformer",
                 path_ranker_type="Flat",
                 gnn_type="Stack",
                 gin_hidden_dim=None,
                 gin_num_layers=1,
                 triplet_margin=1.0,
                 early_stopping_patience=3,
                 early_stopping_metric='val_loss',
                 early_stopping_delta=0.0001,
                 lambda_triplet=0.2,
                 analyze_pruning=False,
                 scheduler_type='plateau', # 新增：調度器類型
                 scheduler_patience=3,     # 新增：ReduceLROnPlateau 的 patience
                 scheduler_factor=0.8,     # 新增：ReduceLROnPlateau 的 factor
                 warmup_steps=500          # 新增：Warmup 的步數
                 ):
        super(Trainer, self).__init__()

        self.tokenizer = tokenizer
        self.encoder = encoder
        self.k_hops = 2 
        self.device = device
        self.cui_vocab_str_to_idx = cui_vocab_str_to_idx 
        self.rev_cui_vocab_idx_to_str = {v: k for k, v in cui_vocab_str_to_idx.items()}
        self.top_n_for_exploration = top_n # ## RENAMED for clarity
        self.finetune_encoder = finetune_encoder
        self.use_soft_labels = use_soft_labels
        self.soft_label_threshold = soft_label_threshold
        self.preserve_gt_in_pruning = preserve_gt_in_pruning
        
        if not self.finetune_encoder:
            print("Encoder parameters are FROZEN.")
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            print("Encoder parameters will be fine-tuned.")
        
        self.loss_type = loss_type
        if self.loss_type.upper() == 'FOCAL':
            print(f"Using Focal Loss with alpha={focal_alpha}, gamma={focal_gamma}")
            # 將實例化的 FocalLoss 賦值給 self.loss_fn_bce
            self.loss_fn_bce = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        elif self.loss_type.upper() == 'BCE':
            print("Using standard BCEWithLogitsLoss")
            self.loss_fn_bce = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"不支援的損失函數類型: {self.loss_type}。請選擇 'BCE' 或 'Focal'。")
        

        self.gmodel = GraphModel(
            g_nx=g_nx, 
            cui_embedding_lookup=cui_embedding_lookup,
            relation_embedding_filepath=relation_embedding_filepath,
            hdim=hdim,
            nums_of_head=nums_of_head,
            num_hops=self.k_hops, # GraphModel 內部可能不需要這個num_hops了，因為Trainer控制跳數
            top_n=self.top_n_for_exploration, # GraphModel 用 top_n 決定下一跳的起始節點
            device=device,
            cui_weights_dict=cui_weights_dict,
            gnn_update=gnn_update,
            path_encoder_type=path_encoder_type,
            path_ranker_type=path_ranker_type,
            gnn_type=gnn_type,
            gin_hidden_dim=gin_hidden_dim,
            gin_num_layers=gin_num_layers
        )

        self.encoder.to(device)
        self.gmodel.to(device)

        self.LR = LR
        self.adam_epsilon = 1e-8
        self.weight_decay = 1e-5

        self.nums_of_epochs = nums_of_epochs
        self.intermediate = intermediate
        self.print_step = 50
        self.distance_metric = distance_metric
        self.mode = 'train'
        self.contrastive_learning = contrastive_learning
        self.triplet_margin = triplet_margin
        self.score_threshold = score_threshold # ## ADDED
        self.lambda_triplet = lambda_triplet
        
        self.save_model_path = save_model_path
        
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric.lower()
        self.early_stopping_delta = early_stopping_delta
        self.epochs_no_improve = 0
        self.early_stop = False
        if self.early_stopping_metric == 'val_loss':
            self.best_metric_val = float('inf')
        elif self.early_stopping_metric == 'val_acc': # 假設我們會有一個主要的 acc 指標
            self.best_metric_val = float('-inf')
        else: # 默認 val_loss
            self.best_metric_val = float('inf')
            print(f"Warning: early_stopping_metric '{early_stopping_metric}' not recognized. Defaulting to 'val_loss'.")
       
        
        self.scheduler_type = scheduler_type
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.warmup_steps = warmup_steps
        
        self.optimizer = None
        self.scheduler = None # 新增 scheduler 屬性
        
        self.analyze_pruning = analyze_pruning
        if self.analyze_pruning:
            print("注意：路徑剪枝分析已啟用，可能會輕微影響性能。")
            self.pruning_stats = {'total_gt_candidates_hop1': 0, 'gt_dropped_hop1': 0, 
                                  'total_gt_candidates_hop2': 0, 'gt_dropped_hop2': 0}
            
            
        print("**** ============= TRAINER (MediQ Tensorized GModel with Thresholding) ============= **** ")
        exp_setting = (f"TRAINER SETUP: SCORE_THRESHOLD: {self.score_threshold}\n"
                       f"INTERMEDIATE LOSS SUPERVISION: {self.intermediate}\n"
                       f"CONTRASTIVE LEARNING: {self.contrastive_learning}\n"
                       f"PATH ENCODER TYPE: {path_encoder_type}\n"
                       f"PATH RANKER TYPE: {path_ranker_type}\n"
                       f"SCHEDULER TYPE: {scheduler_type}\n"
                       f"Finetune Encoder: {self.finetune_encoder}\n"
                       f"Use Softlabel: {self.use_soft_labels}\n"
                       f"Softlabel Threshold: {self.soft_label_threshold}\n"
                       f"Preserve GT Paths: {self.preserve_gt_in_pruning}\n"
                      )
        logging.info(exp_setting)
        print(exp_setting)

    def create_optimizers(self):
                
        no_decay = ["bias", "LayerNorm.weight"]
        main_lr = self.LR
        encoder_lr = 1e-6 # 假設
        weight_decay_val = self.weight_decay # 例如 0.01
        
        gmodel_params_decay = [p for n, p in self.gmodel.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad]
        gmodel_params_no_decay = [p for n, p in self.gmodel.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad]
        
        optimizer_grouped_parameters = [
            {'params': gmodel_params_decay, 'weight_decay': self.weight_decay},
            {'params': gmodel_params_no_decay, 'weight_decay': 0.0}
        ]
        
        if self.finetune_encoder:
            print("Adding encoder parameters to the optimizer.")
            encoder_params_decay = [p for n, p in self.encoder.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad]
            encoder_params_no_decay = [p for n, p in self.encoder.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad]
            
            optimizer_grouped_parameters.extend([
                {'params': encoder_params_decay, 'weight_decay': self.weight_decay, 'lr': encoder_lr},
                {'params': encoder_params_no_decay, 'weight_decay': 0.0, 'lr': encoder_lr}
            ])
        else:
            print("Skipping encoder parameters in the optimizer.")
            
        
        print(f"Using Main Learning Rate: {main_lr}", f" and Encoder Learning Rate: {encoder_lr}")
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=main_lr, eps=self.adam_epsilon)
        print("Optimizer created.")
        


    def compute_context_embedding(self, known_cuis_str_list_sample):
        # (As defined before, ensure it uses self.gmodel.n_encoder_lookup)
        # ...
        if not known_cuis_str_list_sample: return None
        valid_embeddings = []
        unique_known_cuis = list(set(known_cuis_str_list_sample))
        for cui_str in unique_known_cuis:
            # Assuming n_encoder_lookup is the CuiEmbedding object from GraphModel
            if cui_str in self.gmodel.n_encoder_lookup.data: # Check if CUI exists
                 emb = self.gmodel.n_encoder_lookup.encode([cui_str]).to(self.device)
                 valid_embeddings.append(emb.squeeze(0))
        if not valid_embeddings: return None
        return torch.mean(torch.stack(valid_embeddings), dim=0, keepdim=True)


    def _get_gt_indices_tensor(self, gt_cuis_str_list):
        """Converts a list of GT CUI strings to a tensor of global CUI indices."""
        if not gt_cuis_str_list:
            return torch.tensor([], dtype=torch.long, device=self.device)
        gt_indices = [self.cui_vocab_str_to_idx[c_str] for c_str in gt_cuis_str_list if c_str in self.cui_vocab_str_to_idx]
        return torch.tensor(gt_indices, dtype=torch.long, device=self.device)

    def compute_bce_loss_for_hop(self, all_paths_info_hop, gt_cuis_str_list_sample):
        hop_loss = torch.tensor(0.0, device=self.device) # 1. 初始化損失為0

        # 2. 早期退出條件：如果沒有路徑資訊、沒有評分、或者沒有真實標籤，則無法計算損失
        if all_paths_info_hop is None or \
           all_paths_info_hop.get('scores') is None or \
           all_paths_info_hop['scores'].numel() == 0 or \
           not gt_cuis_str_list_sample:
            return hop_loss

        # 3. 提取路徑評分和目標CUI索引
        path_scores = all_paths_info_hop['scores'].squeeze(-1) # [num_paths_this_hop]
        path_target_global_indices = all_paths_info_hop['tgt_idx'] # [num_paths_this_hop] (全局CUI索引)
        
        # 4. 準備真實標籤 (Ground Truth)
        # 將GT CUI字符串列表轉換為全局索引的 Tensor
        gt_indices_tensor = self._get_gt_indices_tensor(gt_cuis_str_list_sample) 
        
        # 5. 聚合評分：針對每個在路徑中出現的 *唯一* 目標CUI，獲取其最高預測評分
        unique_path_targets_global_indices, inverse_indices = torch.unique(
            path_target_global_indices, return_inverse=True
        )

        # 如果沒有唯一路徑目標 (例如，所有路徑都被過濾掉了，或者 cand_tgt_idx_hop 為空導致的)
        if unique_path_targets_global_indices.numel() == 0:
            return hop_loss
        
        if self.use_soft_labels:
            predicted_target_embs = self.gmodel._get_embeddings_by_indices(unique_path_targets_global_indices)
            gt_indices_tensor = self._get_gt_indices_tensor(gt_cuis_str_list_sample)
            if not gt_indices_tensor.numel() > 0:
                # 如果沒有GT，所有標籤都為0
                labels_for_loss = torch.zeros(predicted_target_embs.size(0), device=self.device)
            else:
                gt_target_embs = self.gmodel._get_embeddings_by_indices(gt_indices_tensor)

                # 2. 計算餘弦相似度矩陣
                predicted_target_embs_norm = F.normalize(predicted_target_embs, p=2, dim=1)
                gt_target_embs_norm = F.normalize(gt_target_embs, p=2, dim=1)
                similarity_matrix = torch.matmul(predicted_target_embs_norm, gt_target_embs_norm.t())

                # 3. 為每個預測目標取其與所有GT中的最大相似度
                max_similarity_scores, _ = torch.max(similarity_matrix, dim=1)
                
                # 4. 應用閾值，創建帶有"死區"的軟標籤
                soft_labels = torch.where(
                    max_similarity_scores > self.soft_label_threshold, 
                    max_similarity_scores, 
                    torch.zeros_like(max_similarity_scores)
                )
                labels_for_loss = soft_labels
        else:
            gt_indices_tensor = self._get_gt_indices_tensor(gt_cuis_str_list_sample)
            is_gt_mask = torch.isin(unique_path_targets_global_indices, gt_indices_tensor)
            labels_for_loss = is_gt_mask.float()
        
        
        

        # 使用 torch_scatter (如果可用) 或 fallback 進行聚合
        if TORCH_SCATTER_AVAILABLE: # TORCH_SCATTER_AVAILABLE 應在類或全局定義
            # 確保 inverse_indices 的值在 scatter_max 的有效範圍內
            if unique_path_targets_global_indices.numel() > 0 and \
               path_scores.numel() > 0 and \
               inverse_indices.numel() == path_scores.numel():
                
                # 檢查 inverse_indices 的最大值是否小於 dim_size
                if inverse_indices.max().item() < unique_path_targets_global_indices.size(0):
                    aggregated_scores_for_unique_targets = scatter_max(
                        path_scores, 
                        inverse_indices, 
                        dim=0, 
                        dim_size=unique_path_targets_global_indices.size(0)
                    )[0] # scatter_max 返回 (values, argmax_indices)
                else:
                    # print(f"警告 (BCE scatter_max): inverse_indices 包含越界值。使用迴圈回退。")
                    # 執行迴圈回退邏輯 (與 TORCH_SCATTER_AVAILABLE = False 時相同)
                    temp_aggregated_scores = torch.full((unique_path_targets_global_indices.size(0),), float('-inf'), device=self.device, dtype=path_scores.dtype)
                    for i_loop_idx, unique_tgt_idx_loop in enumerate(unique_path_targets_global_indices):
                        mask_loop = (path_target_global_indices == unique_tgt_idx_loop)
                        if torch.any(mask_loop):
                            temp_aggregated_scores[i_loop_idx] = torch.max(path_scores[mask_loop])
                    aggregated_scores_for_unique_targets = temp_aggregated_scores
            elif unique_path_targets_global_indices.numel() > 0:
                aggregated_scores_for_unique_targets = torch.full((unique_path_targets_global_indices.size(0),), float('-inf'), device=self.device, dtype=path_scores.dtype)
            else: # unique_path_targets_global_indices 為空 (理論上已被上面 numel()==0 捕捉)
                return hop_loss
        else: # Fallback without torch_scatter
            temp_aggregated_scores_list = []
            for unique_tgt_idx_loop in unique_path_targets_global_indices.tolist():
                mask_loop = (path_target_global_indices == unique_tgt_idx_loop)
                if torch.any(mask_loop):
                    temp_aggregated_scores_list.append(torch.max(path_scores[mask_loop]))
                else: # 如果一個 unique target 實際上沒有對應的路徑分數 (不應發生)
                    temp_aggregated_scores_list.append(torch.tensor(float('-inf'), device=self.device, dtype=path_scores.dtype))
            
            if not temp_aggregated_scores_list: return hop_loss
            aggregated_scores_for_unique_targets = torch.stack(temp_aggregated_scores_list)

       
        
        # 7. 計算 BCE 損失
        # 確保 aggregated_scores 和 labels 的形狀和元素數量匹配
        if aggregated_scores_for_unique_targets.numel() > 0 and \
           aggregated_scores_for_unique_targets.size(0) == labels_for_loss.size(0):
             hop_loss = self.loss_fn_bce(aggregated_scores_for_unique_targets, labels_for_loss)
        
            
        return hop_loss


    def compute_triplet_loss_for_hop(self, anchor_embedding, all_paths_info_hop, gt_cuis_str_list_sample):
        triplet_loss = torch.tensor(0.0, device=self.device)
        if anchor_embedding is None or all_paths_info_hop is None or \
           all_paths_info_hop['encoded_embeddings'] is None or \
           all_paths_info_hop['encoded_embeddings'].numel() == 0 or not gt_cuis_str_list_sample:
            return triplet_loss

        # path_embeddings = all_paths_info_hop['encoded_embeddings'] # [num_paths, hdim]
        path_target_global_indices = all_paths_info_hop['tgt_idx']   # [num_paths] (GLOBAL CUI INDICES)

        target_embeddings = self.gmodel._get_embeddings_by_indices(path_target_global_indices)
        
        gt_indices_tensor = self._get_gt_indices_tensor(gt_cuis_str_list_sample)
        if not gt_indices_tensor.numel() > 0: return triplet_loss

        positive_indices_mask = torch.isin(path_target_global_indices, gt_indices_tensor)
        negative_indices_mask = ~positive_indices_mask

        positive_embs = target_embeddings[positive_indices_mask]
        negative_embs = target_embeddings[negative_indices_mask]

        if positive_embs.numel() == 0 or negative_embs.numel() == 0:
            return triplet_loss


        # Simplified Triplet: Anchor vs Random Positive vs Random Negative
        # For more robust triplet loss, you might sample multiple triplets
        # Or use strategies like semi-hard negative mining.
        
        # Select one random positive and one random negative for each anchor
        # Here, anchor is single [1, hdim].
        # We can compare anchor to all positives and all negatives.
        num_pos = positive_embs.size(0)
        num_neg = negative_embs.size(0)

        anchor_expanded_pos = anchor_embedding.expand(num_pos, -1)
        anchor_expanded_neg = anchor_embedding.expand(num_neg, -1)

        if self.distance_metric == "Cosine":
            # We want sim(anchor, pos) to be high, sim(anchor, neg) to be low
            # Loss = margin - (sim_pos_avg - sim_neg_avg)  OR
            # Loss for each pos, neg pair: margin - sim(A,P) + sim(A,N)
            # Let's take average similarity for simplicity here
            sim_pos_all = F.cosine_similarity(anchor_expanded_pos, positive_embs) # [num_pos]
            sim_neg_all = F.cosine_similarity(anchor_expanded_neg, negative_embs) # [num_neg]
            
            # Simplest: average positive similarity vs average negative similarity
            avg_sim_pos = torch.mean(sim_pos_all) if sim_pos_all.numel() > 0 else torch.tensor(0.0, device=self.device)
            avg_sim_neg = torch.mean(sim_neg_all) if sim_neg_all.numel() > 0 else torch.tensor(0.0, device=self.device)
            
            loss_val = self.triplet_margin - avg_sim_pos + avg_sim_neg

        else: # Euclidean distance
            # We want dist(anchor, pos) to be low, dist(anchor, neg) to be high
            # Loss = margin + dist_pos_avg - dist_neg_avg
            dist_pos_all = F.pairwise_distance(anchor_expanded_pos, positive_embs, p=2)
            dist_neg_all = F.pairwise_distance(anchor_expanded_neg, negative_embs, p=2)

            avg_dist_pos = torch.mean(dist_pos_all) if dist_pos_all.numel() > 0 else torch.tensor(0.0, device=self.device)
            avg_dist_neg = torch.mean(dist_neg_all) if dist_neg_all.numel() > 0 else torch.tensor(0.0, device=self.device)

            loss_val = self.triplet_margin + avg_dist_pos - avg_dist_neg
            
        triplet_loss = F.relu(loss_val)
        return triplet_loss


    def forward_per_batch(self, batch):
        input_text_tks_padded = batch['input_text_tks_padded']
        known_cuis_str_batch = batch['known_cuis'] 
        hop1_target_cuis_str_batch = batch['hop1_target_cuis']
        hop2_target_cuis_str_batch = batch['hop2_target_cuis']
        intermediate_target_cuis_batch = batch['intermediate_target_cuis'] # 新增的GT

        input_task_embs_batch = self.encoder(
            input_text_tks_padded['input_ids'].to(self.device),
            input_text_tks_padded['attention_mask'].to(self.device)
        ).pooler_output 

        accumulated_batch_loss = torch.tensor(0.0, device=self.device)
        batch_size = input_task_embs_batch.shape[0]
        
        batch_detailed_predictions = [[] for _ in range(batch_size)]

        for i in range(batch_size): # Sample 迴圈
            sample_loss_this_item = torch.tensor(0.0, device=self.device)
            task_emb_sample = input_task_embs_batch[i].unsqueeze(0) 
            known_cuis_str_sample = known_cuis_str_batch[i]
            context_emb_sample = self.compute_context_embedding(known_cuis_str_sample)
            if context_emb_sample is None: context_emb_sample = task_emb_sample

            current_cui_str_list_for_hop = known_cuis_str_sample
            prev_iter_state_for_next_hop = None
            current_sample_final_preds_dict = {}


            for running_k in range(self.k_hops): # Hop 迴圈
                if not current_cui_str_list_for_hop and running_k > 0 : break
                
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
                               
                all_paths_info_hop, _, \
                next_hop_state_info_for_exploration, stop_flag, debug_info = self.gmodel.one_iteration(
                    task_emb_sample, current_cui_str_list_for_hop, running_k,
                    context_emb_sample, prev_iter_state_for_next_hop,
                    gt_indices_for_pruning=gt_indices_tensor_for_pruning 
                )

                if stop_flag or all_paths_info_hop is None: break

               
                    
                gt_indices_tensor_for_analysis = self._get_gt_indices_tensor(list(set(gt_cuis_str_list_this_hop)))
                gt_cuis_str_list_this_hop = list(set(gt_cuis_str_list_this_hop))
                
                
                if self.analyze_pruning and self.mode == 'eval' and debug_info: # 通常只在驗證集上分析
                    pre_pruning_targets = debug_info.get('pre_pruning_targets')
                    
                    if pre_pruning_targets is not None and pre_pruning_targets.numel() > 0 and gt_indices_tensor_for_analysis.numel() > 0:
                        post_pruning_targets = all_paths_info_hop.get('tgt_idx') if all_paths_info_hop else None
                        
                        if post_pruning_targets is not None:
                            gt_set = set(gt_indices_tensor_for_analysis.tolist())
                            pre_set = set(pre_pruning_targets.tolist())
                            post_set = set(post_pruning_targets.tolist())

                            gt_candidates_before_pruning = gt_set.intersection(pre_set)
                            gt_candidates_after_pruning = gt_set.intersection(post_set)
                            
                            num_gt_candidates = len(gt_candidates_before_pruning)
                            num_gt_dropped = len(gt_candidates_before_pruning - gt_candidates_after_pruning)

                            if running_k == 0:
                                self.pruning_stats['total_gt_candidates_hop1'] += num_gt_candidates
                                self.pruning_stats['gt_dropped_hop1'] += num_gt_dropped
                            elif running_k == 1:
                                self.pruning_stats['total_gt_candidates_hop2'] += num_gt_candidates
                                self.pruning_stats['gt_dropped_hop2'] += num_gt_dropped
                

                current_hop_bce_loss = torch.tensor(0.0, device=self.device)
                current_hop_triplet_loss = torch.tensor(0.0, device=self.device)
                if gt_cuis_str_list_this_hop and all_paths_info_hop.get('scores') is not None and all_paths_info_hop['scores'].numel() > 0:
                    current_hop_bce_loss = self.compute_bce_loss_for_hop(all_paths_info_hop, gt_cuis_str_list_this_hop)
                    if self.contrastive_learning and self.mode == "train" and \
                       all_paths_info_hop.get('encoded_embeddings') is not None and all_paths_info_hop['encoded_embeddings'].numel() > 0:
                        anchor_for_triplet = task_emb_sample
                        current_hop_triplet_loss = self.compute_triplet_loss_for_hop(
                            anchor_for_triplet, all_paths_info_hop, gt_cuis_str_list_this_hop
                        )
                current_hop_total_loss = current_hop_bce_loss + self.lambda_triplet * current_hop_triplet_loss
                
                len_penalty = True
                if len_penalty:
                    if running_k == 0: # running_k=0 代表是第一跳 (1-hop)
                        hop_weighted_loss = 1.0 * current_hop_total_loss
                    elif running_k == 1: # running_k=1 代表是第二跳 (2-hop)
                        hop_weighted_loss = 0.7 * current_hop_total_loss
                else:
                    hop_weighted_loss = current_hop_total_loss
                
                sample_loss_this_item = sample_loss_this_item + hop_weighted_loss
                
                # --- ## MODIFIED: 基於閾值篩選高質量預測，並執行路徑取代 ---
                path_scores = all_paths_info_hop['scores'].squeeze(-1) # [num_paths]
                normalized_scores = torch.sigmoid(path_scores)
                confident_mask = (normalized_scores >= self.score_threshold)               
                confident_path_indices = torch.where(confident_mask)[0]

                if confident_path_indices.numel() > 0:
                    conf_orig_srcs = all_paths_info_hop['mem_orig_src_idx'][confident_path_indices]
                    conf_first_edges = all_paths_info_hop['mem_first_edge_idx'][confident_path_indices] if all_paths_info_hop['mem_first_edge_idx'] is not None else None
                    conf_hop_srcs = all_paths_info_hop['src_idx'][confident_path_indices] # 當前跳的源 (中間節點 for hop2)
                    conf_hop_edges = all_paths_info_hop['edge_idx'][confident_path_indices] # 當前跳的邊 (Rel2 for hop2)
                    conf_hop_tgts = all_paths_info_hop['tgt_idx'][confident_path_indices]   # 當前跳的目標 (最終目標 for hop2)
                    conf_scores = normalized_scores[confident_path_indices]

                    for k_path in range(confident_path_indices.numel()):
                        orig_s_idx = conf_orig_srcs[k_path].item()
                        inter_s_idx = conf_hop_srcs[k_path].item() # 中間節點 (如果是第二跳) 或最終目標 (如果是第一跳的源)
                        final_t_idx = conf_hop_tgts[k_path].item()
                        current_score = conf_scores[k_path].item()
                        
                        r1_idx = conf_first_edges[k_path].item() if conf_first_edges is not None and conf_first_edges[k_path].item() != -1 else -1
                        r2_idx = conf_hop_edges[k_path].item() # 當前跳的邊

                        if running_k == 0: # 處理第一跳的高質量預測
                            # 路徑是: orig_s --r2(當前邊)--> final_t
                            path_key = (orig_s_idx, final_t_idx) # 用 (最初始源, 1跳目標) 作為鍵
                            # full_path_tuple = (orig_s_idx, r2_idx, final_t_idx, -1, -1) # Rel1, Inter, Rel2, Final
                            # 這裡 orig_s_idx == inter_s_idx， r1_idx == -1
                            full_path_tuple_for_pred = (orig_s_idx, r2_idx, final_t_idx)

                            if path_key not in current_sample_final_preds_dict or \
                               current_score > current_sample_final_preds_dict[path_key]['score']:
                                current_sample_final_preds_dict[path_key] = {
                                    "path_tuple_indices": full_path_tuple_for_pred,
                                    "score": current_score,
                                    "hop": 1,
                                    "final_target_idx": final_t_idx
                                }
                        elif running_k == 1: # 處理第二跳的高質量預測
                            # 路徑是: orig_s --r1--> inter_s --r2--> final_t
                            first_hop_key = (orig_s_idx, inter_s_idx) # 檢查這個1-hop路徑是否已存在
                            full_path_tuple_for_pred = (orig_s_idx, r1_idx, inter_s_idx, r2_idx, final_t_idx)

                            # 如果這個2-hop路徑的1-hop部分已經是一個高質量預測，
                            # 且當前2-hop路徑分數更高，則取代；或者如果1-hop部分不是高質量預測，則直接添加2-hop。
                            # (簡化邏輯: 如果存在對應的1-hop，且新2-hop更好，則取代。否則，考慮作為新路徑添加)
                            
                            # 優先2-hop路徑
                            # 我們用 (orig_s, inter_s) 作為key來查找是否可以被取代的1-hop路徑
                            # 或者，如果我們用 (orig_s, final_t) for 1-hop / (orig_s, inter_s, final_t) for 2-hop as key for uniqueness
                            # 這裡的邏輯是：如果一個 (orig_s, inter_s) 的1-hop路徑存在，現在有一個從它擴展的2-hop路徑，則取代。
                            
                            if first_hop_key in current_sample_final_preds_dict and \
                               current_sample_final_preds_dict[first_hop_key]['hop'] == 1:
                                # 發現了可以被擴展的1-hop路徑，用當前的2-hop路徑取代它
                                # (如果分數更高，或者無條件取代更長的路徑)
                                # 這裡我們無條件用更長的路徑取代，並更新分數
                                del current_sample_final_preds_dict[first_hop_key] # 刪除舊的1-hop
                                # 新的鍵可以用 (orig_s, final_t) 加上 hop=2 標記，或者用完整的 (orig_s, inter_s, final_t)
                                # 為了唯一性，我們用 (orig_s, inter_s, final_t) 作为2-hop的key
                                two_hop_path_key = (orig_s_idx, inter_s_idx, final_t_idx)
                                current_sample_final_preds_dict[two_hop_path_key] = {
                                     "path_tuple_indices": full_path_tuple_for_pred,
                                     "score": current_score,
                                     "hop": 2,
                                     "final_target_idx": final_t_idx
                                }
                            else:
                                # 沒有可取代的1-hop路徑，或者 first_hop_key 已經是個2-hop路徑
                                # 將此2-hop路徑作為一個新的獨立預測加入（如果鍵不存在或分數更高）
                                two_hop_path_key = (orig_s_idx, inter_s_idx, final_t_idx) # 確保key的唯一性
                                if two_hop_path_key not in current_sample_final_preds_dict or \
                                   current_score > current_sample_final_preds_dict[two_hop_path_key]['score']:
                                    current_sample_final_preds_dict[two_hop_path_key] = {
                                        "path_tuple_indices": full_path_tuple_for_pred,
                                        "score": current_score,
                                        "hop": 2,
                                        "final_target_idx": final_t_idx
                                    }
                
                # --- 更新下一跳的狀態 (仍然基於 top_n 進行探索) ---
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
                
                
            detailed_preds_for_sample = []
            for pred_info in current_sample_final_preds_dict.values():
                target_cui_str = self.rev_cui_vocab_idx_to_str.get(pred_info['final_target_idx'])
                if target_cui_str:
                    detailed_preds_for_sample.append({
                        "target_cui": target_cui_str,
                        "score": pred_info['score'],
                        "hop": pred_info['hop'],
                        "path_indices": pred_info['path_tuple_indices']
                    })
            batch_detailed_predictions[i] = detailed_preds_for_sample
            
            accumulated_batch_loss += sample_loss_this_item


        avg_batch_loss = accumulated_batch_loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=self.device)
        
        # final_predicted_cuis_str_batch 用於與 hop2_target_cuis 比較 (或者一個合併的GT)
        # batch_hop1_predictions_for_acc 用於與 hop1_target_cuis 比較
        # 這個返回值需要調整，以包含不同hop的預測
        return avg_batch_loss, batch_detailed_predictions


    def measure_accuracy(self, batch_detailed_predictions, batch_gts):
        """
        接收詳細的預測結果和批次GT，計算1跳、2跳和最終預測的 Precision, Recall, F1。
        
        Args:
            batch_detailed_predictions (list[list[dict]]): forward_per_batch 返回的結構化預測。
            batch_gts (dict): 來自 DataLoader 的原始批次數據，包含所有GT列表。
        
        Returns:
            dict: 一個包含所有計算出的指標的字典。
        """
        batch_size = len(batch_detailed_predictions)
        if batch_size == 0:
            return {
                'final': {'p': 0, 'r': 0, 'f1': 0},
                'hop1': {'p': 0, 'r': 0, 'f1': 0},
                'hop2': {'p': 0, 'r': 0, 'f1': 0},
            }

        # 準備用於聚合的列表
        list_final_preds, list_hop1_preds, list_hop2_preds = [], [], []
        list_final_gts, list_hop1_gts, list_hop2_gts = [], [], []

        # 1. 遍歷批次中的每個樣本，拆分預測和GT
        for i in range(batch_size):
            sample_detailed_preds = batch_detailed_predictions[i]
            
            # --- 分離預測集 (基於 'hop' 標籤) ---
            # 最終預測是所有跳數的並集
            final_preds_for_sample = {p['target_cui'] for p in sample_detailed_preds}
            # 存活下來的1跳預測
            hop1_preds_for_sample = {p['target_cui'] for p in sample_detailed_preds if p['hop'] == 1}
            # 2跳預測
            hop2_preds_for_sample = {p['target_cui'] for p in sample_detailed_preds if p['hop'] == 2}
            
            list_final_preds.append(final_preds_for_sample)
            list_hop1_preds.append(hop1_preds_for_sample)
            list_hop2_preds.append(hop2_preds_for_sample)

            # --- 分離GT集 ---
            hop1_gt_for_sample = set(batch_gts['hop1_target_cuis'][i])
            hop2_gt_for_sample = set(batch_gts['hop2_target_cuis'][i])
            combined_gt_for_sample = hop1_gt_for_sample.union(hop2_gt_for_sample)

            list_final_gts.append(combined_gt_for_sample)
            list_hop1_gts.append(hop1_gt_for_sample)
            list_hop2_gts.append(hop2_gt_for_sample)

        # 2. 為每個類別計算指標
        def _calculate_prf1(pred_sets, gold_sets):
            all_p, all_r, all_f1 = [], [], []
            for i in range(len(pred_sets)):
                preds = pred_sets[i]
                golds = gold_sets[i]
                
                num_pred = len(preds)
                num_gold = len(golds)
                if num_gold == 0: # 如果這個樣本沒有GT，則不參與該項指標計算
                    continue 

                num_intersect = len(golds.intersection(preds))
                
                p = num_intersect / num_pred if num_pred > 0 else 0.0
                r = num_intersect / num_gold if num_gold > 0 else 0.0
                f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
                all_p.append(p); all_r.append(r); all_f1.append(f1)
            
            avg_p = np.mean(all_p) if all_p else 0.0
            avg_r = np.mean(all_r) if all_r else 0.0
            avg_f1 = np.mean(all_f1) if all_f1 else 0.0
            return {'p': avg_p, 'r': avg_r, 'f1': avg_f1}

        metrics_final = _calculate_prf1(list_final_preds, list_final_gts)
        metrics_hop1 = _calculate_prf1(list_hop1_preds, list_hop1_gts)
        metrics_hop2 = _calculate_prf1(list_hop2_preds, list_hop2_gts)
        
        return {
            'final': metrics_final,
            'hop1': metrics_hop1,
            'hop2': metrics_hop2
        }

    def train(self, train_data_loader, dev_data_loader, lr_scheduler=None):
        if self.optimizer is None: self.create_optimizers()
        update_step = 8 
        
        if self.scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min' if self.early_stopping_metric == 'val_loss' else 'max',
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                # verbose=True
            )
            print(f"ReduceLROnPlateau scheduler created: mode='{'min' if self.early_stopping_metric == 'val_loss' else 'max'}', patience={self.scheduler_patience}, factor={self.scheduler_factor}.")
        elif self.scheduler_type == 'warmup':
            num_training_steps = len(train_data_loader) * self.nums_of_epochs
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=num_training_steps
            )
            print(f"Linear scheduler with warmup created: warmup_steps={self.warmup_steps}, total_steps={num_training_steps}.")
        else:
            print("No learning rate scheduler will be used.")
            self.scheduler = None

        for ep in range(self.nums_of_epochs):
            print(f"\n--- Starting Epoch {ep+1}/{self.nums_of_epochs} ---")
            self.mode = 'train'; self.encoder.train(); self.gmodel.train()
            epoch_loss_train_list = []
            epoch_p1_train, epoch_r1_train, epoch_f1_1_train = [], [], [] # For 1-hop
            epoch_p2_train, epoch_r2_train, epoch_f1_2_train = [], [], [] # For 2-hop
            epoch_p_final_train, epoch_r_final_train, epoch_f1_final_train = [], [], [] # For final preds

            accumulated_loss_for_step = torch.tensor(0.0, device=self.device)
            train_pbar = tqdm(train_data_loader, desc=f"Epoch {ep+1} Training")
            batch_idx_in_epoch = 0

            for batch in train_pbar:
                if batch is None: continue 
                
                batch_avg_loss, batch_detailed_predictions = self.forward_per_batch(batch)
                            
                batch_metrics = self.measure_accuracy(batch_detailed_predictions, batch)
                
                # 記錄用於epoch平均的指標
                epoch_p1_train.append(batch_metrics['hop1']['p']); epoch_r1_train.append(batch_metrics['hop1']['r']); epoch_f1_1_train.append(batch_metrics['hop1']['f1'])
                epoch_p2_train.append(batch_metrics['hop2']['p']); epoch_r2_train.append(batch_metrics['hop2']['r']); epoch_f1_2_train.append(batch_metrics['hop2']['f1'])
                epoch_p_final_train.append(batch_metrics['final']['p']); epoch_r_final_train.append(batch_metrics['final']['r']); epoch_f1_final_train.append(batch_metrics['final']['f1'])

                if torch.isnan(batch_avg_loss) or torch.isinf(batch_avg_loss):
                    print(f"Warning: NaN/Inf loss @ Epoch {ep+1}, Batch {batch_idx_in_epoch+1}. Skipping."); self.optimizer.zero_grad(); accumulated_loss_for_step = torch.tensor(0.0, device=self.device); batch_idx_in_epoch +=1; continue
                
                
                loss_to_accumulate = batch_avg_loss / update_step
                # print(f"DEBUG: loss_to_accumulate: {loss_to_accumulate.item()}, requires_grad: {loss_to_accumulate.requires_grad}, grad_fn: {loss_to_accumulate.grad_fn}")
                # print(f"DEBUG: batch_avg_loss: {batch_avg_loss.item()}, requires_grad: {batch_avg_loss.requires_grad}, grad_fn: {batch_avg_loss.grad_fn}")
        
                loss_to_accumulate.backward()
                accumulated_loss_for_step += loss_to_accumulate.detach()
                epoch_loss_train_list.append(batch_avg_loss.item())

                if (batch_idx_in_epoch + 1) % update_step == 0:
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.gmodel.parameters(), 1.0)
                    self.optimizer.step(); 
                    if self.scheduler and self.scheduler_type == 'warmup':
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    train_pbar.set_postfix({'L': f'{accumulated_loss_for_step.item():.3f}', 
                                            'F1@1': f'{batch_metrics["hop1"]["f1"]:.3f}','F1@2': f'{batch_metrics["hop2"]["f1"]:.3f}', 'F1@Final': f'{batch_metrics["final"]["f1"]:.3f}'})
                    accumulated_loss_for_step = torch.tensor(0.0, device=self.device)
                batch_idx_in_epoch +=1
                
            # Process the last batch
            if batch_idx_in_epoch % update_step != 0 and accumulated_loss_for_step.item() > 0 :
                 torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                 torch.nn.utils.clip_grad_norm_(self.gmodel.parameters(), 1.0)
                 self.optimizer.step(); 
                 if self.scheduler and self.scheduler_type == 'warmup':
                    self.scheduler.step()
                 self.optimizer.zero_grad()
                 

            avg_ep_train_loss = np.mean(epoch_loss_train_list) if epoch_loss_train_list else float('nan')
            avg_ep_f1_1_train = np.mean(epoch_f1_1_train) if epoch_f1_1_train else float('nan')
            avg_ep_f1_2_train = np.mean(epoch_f1_2_train) if epoch_f1_2_train else float('nan')
            avg_ep_f1_final_train = np.mean(epoch_f1_final_train) if epoch_f1_final_train else float('nan')
            print(f"\nEpoch {ep+1} Train Avg: Loss={avg_ep_train_loss:.4f}, F1@1={avg_ep_f1_1_train:.4f}, F1@2={avg_ep_f1_2_train:.4f}, F1@Final={avg_ep_f1_final_train:.4f}")

            avg_ep_dev_loss, avg_ep_p1_dev, avg_ep_r1_dev, avg_ep_f1_1_dev, \
            avg_ep_p2_dev, avg_ep_r2_dev, avg_ep_f1_2_dev, \
            avg_ep_p_final_dev, avg_ep_r_final_dev, avg_ep_f1_final_dev = self.validate(dev_data_loader)
            print(f"Epoch {ep+1} Valid Avg: Loss={avg_ep_dev_loss:.4f} \n P@1={avg_ep_p1_dev:.4f}, R@1={avg_ep_r1_dev:.4f}, F1@1={avg_ep_f1_1_dev:.4f} \n P@2={avg_ep_p2_dev:.4f}, R@2={avg_ep_r2_dev:.4f}, F1@2={avg_ep_f1_2_dev:.4f} \n P@Final={avg_ep_p_final_dev:.4f}, R@Final={avg_ep_r_final_dev:.4f}, F1@Final={avg_ep_f1_final_dev:.4f}")
            
            if self.scheduler and self.scheduler_type == 'plateau':
                # ReduceLROnPlateau 需要一個監控指標來決定是否降低學習率
                metric_for_scheduler = avg_ep_dev_loss if self.early_stopping_metric == 'val_loss' else avg_ep_f1_final_dev
                self.scheduler.step(metric_for_scheduler)
            
            if lr_scheduler: lr_scheduler.step()

            # ## MODIFIED: Early Stopping and Model Saving based on a primary metric, e.g., F1@Final
            if self.early_stopping_metric == 'val_loss':
                current_metric_val = avg_ep_dev_loss
            elif self.early_stopping_metric == 'val_acc':
                current_metric_val = avg_ep_f1_final_dev
          
            improved = False
            if self.early_stopping_metric == 'val_loss':
                if current_metric_val < self.best_metric_val - self.early_stopping_delta: improved = True
            elif self.early_stopping_metric == 'val_acc': # 這裡用 F1@Final 作為 val_acc
                if current_metric_val > self.best_metric_val + self.early_stopping_delta: improved = True
            
            if improved:
                print(f"Validation metric ({self.early_stopping_metric}) improved ({self.best_metric_val:.4f} --> {current_metric_val:.4f}). Saving model...")
                self.best_metric_val = current_metric_val; self.epochs_no_improve = 0
                if self.save_model_path:
                    try:
                        torch.save(self.gmodel.state_dict(), self.save_model_path)
                        encoder_save_path = os.path.join(os.path.dirname(self.save_model_path), "encoder.pth")
                        torch.save(self.encoder.state_dict(), encoder_save_path)
                        print(f"Model saved to {self.save_model_path} and {encoder_save_path}")
                    except Exception as e: print(f"Error saving model: {e}")
            else:
                self.epochs_no_improve += 1
                print(f"Validation metric ({self.early_stopping_metric}) did not improve for {self.epochs_no_improve} epoch(s). Best: {self.best_metric_val:.4f}")

            if self.epochs_no_improve >= self.early_stopping_patience:
                self.early_stop = True; print(f"\nEarly stopping triggered after {ep+1} epochs."); break 
            print("-" * 50)
        
        if not self.early_stop: print("Training finished after all epochs.")


    def validate(self, dev_data_loader):
        print("Running validation...")
        self.mode = 'eval'; self.encoder.eval(); self.gmodel.eval()
        epoch_loss_dev_list = []
        epoch_p1_dev, epoch_r1_dev, epoch_f1_1_dev = [], [], []
        epoch_p2_dev, epoch_r2_dev, epoch_f1_2_dev = [], [], []
        epoch_p_final_dev, epoch_r_final_dev, epoch_f1_final_dev = [], [], []
        
        dev_pbar = tqdm(dev_data_loader, desc="Validation")
        
        
        if self.analyze_pruning:
            self.pruning_stats = {'total_gt_candidates_hop1': 0, 'gt_dropped_hop1': 0, 
                                  'total_gt_candidates_hop2': 0, 'gt_dropped_hop2': 0}
        
        with torch.no_grad():
            for batch in dev_pbar:
                if batch is None: continue
                batch_avg_loss, batch_detailed_predictions = self.forward_per_batch(batch)
                
                batch_metrics = self.measure_accuracy(batch_detailed_predictions, batch)

                epoch_loss_dev_list.append(batch_avg_loss.item())
                epoch_p1_dev.append(batch_metrics['hop1']['p']); epoch_r1_dev.append(batch_metrics['hop1']['r']); epoch_f1_1_dev.append(batch_metrics['hop1']['f1'])
                epoch_p2_dev.append(batch_metrics['hop2']['p']); epoch_r2_dev.append(batch_metrics['hop2']['r']); epoch_f1_2_dev.append(batch_metrics['hop2']['f1'])
                epoch_p_final_dev.append(batch_metrics['final']['p']); epoch_r_final_dev.append(batch_metrics['final']['r']); epoch_f1_final_dev.append(batch_metrics['final']['f1'])
                dev_pbar.set_postfix({'L': f'{batch_avg_loss.item():.3f}', 'F1@1': f'{batch_metrics["hop1"]["f1"]:.3f}','F1@2': f'{batch_metrics["hop2"]["f1"]:.3f}', 'F1@Final': f'{batch_metrics["final"]["f1"]:.3f}'})

        avg_loss = np.mean(epoch_loss_dev_list) if epoch_loss_dev_list else float('nan')
        avg_p1 = np.mean(epoch_p1_dev) if epoch_p1_dev else 0.0
        avg_r1 = np.mean(epoch_r1_dev) if epoch_r1_dev else 0.0
        avg_f1_1 = np.mean(epoch_f1_1_dev) if epoch_f1_1_dev else 0.0
        avg_p2 = np.mean(epoch_p2_dev) if epoch_p2_dev else 0.0
        avg_r2 = np.mean(epoch_r2_dev) if epoch_r2_dev else 0.0
        avg_f1_2 = np.mean(epoch_f1_2_dev) if epoch_f1_2_dev else 0.0
        avg_p_final = np.mean(epoch_p_final_dev) if epoch_p_final_dev else 0.0
        avg_r_final = np.mean(epoch_r_final_dev) if epoch_r_final_dev else 0.0
        avg_f1_final = np.mean(epoch_f1_final_dev) if epoch_f1_final_dev else 0.0
        
        
        if self.analyze_pruning:
            print("\n--- Path Pruning Analysis Results ---")
            total_hop1 = self.pruning_stats['total_gt_candidates_hop1']
            dropped_hop1 = self.pruning_stats['gt_dropped_hop1']
            percent_dropped_hop1 = (dropped_hop1 / total_hop1 * 100) if total_hop1 > 0 else 0
            print(f"Hop 1: Dropped {dropped_hop1} of {total_hop1} GT candidates ({percent_dropped_hop1:.2f}%) due to pruning.")
            
            total_hop2 = self.pruning_stats['total_gt_candidates_hop2']
            dropped_hop2 = self.pruning_stats['gt_dropped_hop2']
            percent_dropped_hop2 = (dropped_hop2 / total_hop2 * 100) if total_hop2 > 0 else 0
            print(f"Hop 2: Dropped {dropped_hop2} of {total_hop2} GT candidates ({percent_dropped_hop2:.2f}%) due to pruning.")
            print("-------------------------------------\n")       
       

        return avg_loss, avg_p1, avg_r1, avg_f1_1,avg_p2, avg_r2, avg_f1_2, avg_p_final, avg_r_final, avg_f1_final

# ====================== Main Block ======================
if __name__ =='__main__':

    

    # --- 常規設置與資源加載 (假設這些已存在或從 args 讀取) ---
    # !!! 確保這些變量已定義並加載好 !!!
    # args = parser.parse_args() # 如果使用 argparse
    TEST_TOKENIZER_PATH = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext" # 或者您的本地路徑
    GRAPH_NX_FILE = "./drknows/SNOMED_CUI_MAJID_Graph_wSelf.pkl" # 原始 NetworkX 圖
    
    CUI_EMBEDDING_FILE = "./drknows/GraphModel_SNOMED_CUI_Embedding.pkl"
    RELATION_EMBEDDING_FILE = "./drknows/relation_sapbert_embeddings.pkl"
    # TRAIN_ANNOTATION_FILE = "./MediQ/mediq_train_preprocessed.jsonl"
    # DEV_ANNOTATION_FILE = './MediQ/mediq_dev_preprocessed.jsonl'
    TRAIN_ANNOTATION_FILE = "./MediQ/mediq_train_annotations_bm25_20.json"
    DEV_ANNOTATION_FILE = './MediQ/mediq_dev_annotations_bm25_20.json'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(TEST_TOKENIZER_PATH)
    base_encoder_model = AutoModel.from_pretrained(TEST_TOKENIZER_PATH) # Will be moved to device in Trainer
    
    # Load NetworkX graph first
    try:
        g_nx_loaded = pickle.load(open(GRAPH_NX_FILE, "rb"))
        print(f"NetworkX graph loaded successfully from {GRAPH_NX_FILE}")
    except Exception as e:
        print(f"Error loading NetworkX graph: {e}")
        exit()
        
    try:
        cui_embedding_lookup_obj = CuiEmbedding(CUI_EMBEDDING_FILE, device=device)
        print(" CuiEmbedding 實例化成功。")
    except Exception as e:
        print(f"實例化 CuiEmbedding 時出錯: {e}"); exit()
        
        

    

    
    print("從已加載的圖譜創建 CUI string to global index mapping for Trainer...")
    _nodes_for_vocab = sorted(list(g_nx_loaded.nodes()))
    cui_vocab_for_trainer = {cui_str: i for i, cui_str in enumerate(_nodes_for_vocab)}
    print(f"Trainer's cui_vocab_str_to_idx created with {len(cui_vocab_for_trainer)} entries.")

    


    # Hyperparameters
    hdim = base_encoder_model.config.hidden_size
    nums_of_head = 3 
    top_n = 8 
    epochs = 100 
    LR = 1e-5
    intermediate_loss_flag = True 
    contrastive_flag = False
    batch_size = 2 
    
    gin_hidden_dim_val = hdim 
    gin_num_layers_val = 2  


    temp_edge_enc = EdgeOneHot(graph=g_nx_loaded, device=device)
    actual_input_edge_dim_for_gin = temp_edge_enc.onehot_mat.shape[-1]
    if actual_input_edge_dim_for_gin == 0 and g_nx_loaded.number_of_edges() > 0:
        print("警告: 為 GIN 確定的 input_edge_dim 是 0。GIN 可能無法按預期工作，如果它需要邊緣特徵。")
        
    save_model_dir = "./saved_models_mediq" # 改個目錄名
    if not os.path.exists(save_model_dir): os.makedirs(save_model_dir)
    model_save_path = os.path.join(save_model_dir, "gmodel_mediq_best.pth")

    print("Instantiating Trainer with Tensorized GraphModel call and REAL Embeddings...")
    trainer_instance = Trainer(
        tokenizer=tokenizer,
        encoder=base_encoder_model,
        g_nx=g_nx_loaded,
        cui_embedding_lookup=cui_embedding_lookup_obj,
        relation_embedding_filepath=RELATION_EMBEDDING_FILE, 
        hdim=hdim,
        nums_of_head=nums_of_head,
        cui_vocab_str_to_idx=cui_vocab_for_trainer,
        top_n=top_n,
        device=device,
        nums_of_epochs=epochs, 
        LR=LR,
        loss_type="FOCAL",
        focal_alpha=0.95,
        focal_gamma=2.0,
        cui_weights_dict=None, 
        contrastive_learning=contrastive_flag,
        intermediate=intermediate_loss_flag,
        score_threshold=0.5,
        finetune_encoder=False,        
        use_soft_labels=False,         
        soft_label_threshold=0.8,     
        preserve_gt_in_pruning=True,         
        save_model_path=model_save_path,
        gnn_update=True, 
        path_encoder_type="Transformer",
        path_ranker_type="Flat",
        gnn_type="GAT", 
        gin_hidden_dim=gin_hidden_dim_val,
        gin_num_layers=gin_num_layers_val,
        early_stopping_patience=3,
        early_stopping_metric='val_loss',
        early_stopping_delta=0.0001,
        lambda_triplet=0.01,
        analyze_pruning=False,
        scheduler_type="warmup",
        warmup_steps=2000
    )
    print("Trainer instantiated.")

    print("\nCreating optimizer...")
    trainer_instance.create_optimizers()
    lr_scheduler_instance = None 

    print("\nLoading datasets...")
    try:
        # train_dataset_obj = MediQPreprocessedDataset(TRAIN_ANNOTATION_FILE)
        # dev_dataset_obj = MediQPreprocessedDataset(DEV_ANNOTATION_FILE)
        train_dataset_obj = MediQAnnotatedDataset(TRAIN_ANNOTATION_FILE, tokenizer)
        dev_dataset_obj = MediQAnnotatedDataset(DEV_ANNOTATION_FILE, tokenizer)
    except Exception as e:
        print(f"Error loading datasets: {e}"); exit()
        
    if len(train_dataset_obj) == 0 or len(dev_dataset_obj) == 0:
        print("Error: A dataset is empty after loading!"); exit()
        
        

    # train_loader_instance = DataLoader(train_dataset_obj, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_mediq_preprocessed, num_workers=6, pin_memory=True)
    # dev_loader_instance = DataLoader(dev_dataset_obj, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_mediq_preprocessed, num_workers=6, pin_memory=True)
    train_loader_instance = DataLoader(train_dataset_obj, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_mediq_paths, num_workers=6, pin_memory=True)
    dev_loader_instance = DataLoader(dev_dataset_obj, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_mediq_paths, num_workers=6, pin_memory=True)
    print("Dataloaders created.")

    print("\n" + "="*30 + "\n STARTING Tensorized Trainer RUN  \n" + "="*30 + "\n")
    try: 
        trainer_instance.train(train_loader_instance, dev_loader_instance, lr_scheduler_instance)
    except Exception as e:
        print(f"ERROR DURING TRAINING RUN: {e}")
        import traceback
        traceback.print_exc()
    print("\n" + "="*30 + "\n TENSORIZED TRAINER RUN FINISHED \n" + "="*30 + "\n")

    
    

