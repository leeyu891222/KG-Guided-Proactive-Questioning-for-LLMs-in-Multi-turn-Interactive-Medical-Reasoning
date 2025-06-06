import unittest
import os
import json
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import networkx as nx
import numpy as np
from unittest.mock import MagicMock, patch

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import time
import torch
#from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math 
from transformers import AutoTokenizer, AutoModel # Assuming tokenizer is loaded elsewhere or passed in
import pandas as pd 
import json
from tqdm import tqdm
import argparse
import logging
import datetime
import random
import re
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import networkx as nx
#from mpi4py import MPI
from collections import OrderedDict, defaultdict


# Placeholder for a tokenizer if not loaded globally
# tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext") # Or your model path
# Try to import torch_scatter, if not available, print a warning
try:
    from torch_scatter import scatter_add, scatter_max
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
        # ... __init__ 的前半部分與之前相同，無需修改 ...
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
        
        # ## MODIFIED: 創建三個獨立的集合來儲存不同類型的 GT
        hop1_target_cuis_set = set()
        hop2_target_cuis_set = set()
        intermediate_target_cuis_set = set() # ## ADDED
        
        for k_idx in known_indices:
            for u_idx in unknown_indices:
                path_key_optional = f"{k_idx}_{u_idx}"
                if path_key_optional in paths_between_facts:
                    for path_data in paths_between_facts[path_key_optional]:
                        if not path_data or not isinstance(path_data, list): continue
                        path_len = len(path_data)
                        target_cui_in_path = path_data[-1]
                        is_valid_cui = isinstance(target_cui_in_path, str) and target_cui_in_path.startswith('C')
                        
                        if is_valid_cui and isinstance(facts_cuis[u_idx], list) and \
                           target_cui_in_path in facts_cuis[u_idx]:
                            
                            # ## MODIFIED: 根據路徑長度將目標 CUI 分配到正確的集合
                            if path_len == 3: # 1-hop path
                                hop1_target_cuis_set.add(target_cui_in_path)
                            elif path_len == 5: # 2-hop path
                                intermediate_cui = path_data[2] # 中間節點
                                intermediate_target_cuis_set.add(intermediate_cui) # ## ADDED
                                hop2_target_cuis_set.add(target_cui_in_path) # 最終目標
        
        # ## MODIFIED: 修改返回 None 的條件。
        # 現在，一個樣本被視為無效，如果它連一個 1-hop 或 2-hop 的 GT 都沒有。
        # 訓練器將決定如何使用這些GT。例如，如果只訓練2-hop，那麼 hop2_target_cuis 為空就可能是個問題。
        # 為了通用性，我們只要至少有一個GT就返回。
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
            "intermediate_target_cuis": list(intermediate_target_cuis_set), # ## ADDED
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


# ====================== gnn_utils ===================
# Graph utils functions 
def retrieve_cuis(text,g, matcher):
    # Retrieve cuis from quickUMLS 
    output = matcher.match(text)
    #output
    cui_output= [ii['cui'] for i in output for ii in i if ii['cui'] in g.nodes]
    terms = [ii['term'] for i in output for ii in i if ii['cui'] in g.nodes]
    cui_outputs = set(cui_output)

    # answer: C0010346 
    return cui_outputs, output

def retrieve_subgraphs(cuis, g):
    # Get subgraphs into a dictionary 
    paths = {}
    for c in cuis:
        paths[c] = [] 
        nodes = list(g.neighbors(c))
        for n in nodes:
            edge_label = g.get_edge_data(c, n)['label']
            paths[c].append([n, edge_label])
    return paths 


def retrieve_phrases(paths, cui_aui_mappings):
    # Map CUI back to phrases to get representation  
    phrase_paths = {}
    for s, t in paths.items():
        sp = cui_aui_mappings[s][0][1] 
        phrase_paths[sp] = []
        for tn in t:
            vp = cui_aui_mappings[tn[0]][0][1]
            phrase_paths[sp].append([vp, tn[1]])
    return phrase_paths





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




# Graph retriever utils 
def project_cui_to_vocab(all_paths_df, cui_vocab):
    vocab_idx = []
    new_srcs = all_paths_df['Tgt']
    for _ in new_srcs:
        vocab_idx.append(cui_vocab[_])
    return vocab_idx 


def sort_visited_paths(indices, all_paths_df, visited_path_embs, prev_visited_paths):
    # Postprocess for top-n selected CUIs
    visited_paths = {}
    new_src_cuis_emb = {}
    if len(prev_visited_paths) == 0:
        for _ in indices:
            k = _[0].item() 
            new_src = all_paths_df.iloc[k]['Tgt'] 
            p = all_paths_df.iloc[k]['Src'] + " --> " + all_paths_df.iloc[k]['Edge'] + " --> " + new_src
            visited_paths[new_src] = p # for explainability
            new_src_cuis_emb[new_src] = visited_path_embs[_[0],:] # src CUI embedding to compute next iteration paths
    else:
        for _ in indices:
            k = _[0].item() # index of the top-n path 
            new_src = all_paths_df.iloc[k]['Tgt'] 
            if all_paths_df.iloc[k]['Src'] in prev_visited_paths:
                prev_p = prev_visited_paths[all_paths_df.iloc[k]['Src']]
                p = prev_p +" --> " + all_paths_df.iloc[k]['Edge'] + " --> " + new_src 
            else:
                p = all_paths_df.iloc[k]['Src'] + " --> " + all_paths_df.iloc[k]['Edge'] + " --> " + new_src
            visited_paths[new_src] = p # for explainability
            new_src_cuis_emb[new_src] = visited_path_embs[_[0],:] 

    return visited_paths, new_src_cuis_emb 

def prune_paths(input_text_vec, cand_neighbors_vs, cand_neighbors_list, threshold=0.8):
    """Purpose: filter out the target CUIs that are not 
    """
    orig_index = len(cand_neighbors_list) 
    tgt_embs = cand_neighbors_vs.detach().numpy()
    xq = input_text_vec.clone().cpu().detach().numpy() # clone the task embedding 
    new_cand_neighbors_lists = [] 
    d = tgt_embs.shape[-1]
    nb = tgt_embs.shape[0]
    nq = 1
    k =int(nb*threshold) # sample top K nodes with similarity 
    #index = faiss.IndexFlatL2(d)   # build the index for euclidean distance 
    index=faiss.IndexFlatIP(d)     # build the index for cosine distance 
    index.add(tgt_embs)                  # add vectors to the index
    D, I = index.search(xq, k)     # actual search, return distance and index 
    new_cand_neighbor_vs = []
    I_sorted = np.sort(I, axis=1)
    new_cand_neighbor_vs = tgt_embs[I_sorted[0]]
    #print(new_cand_neighbor_vs.shape)
    new_cand_neighbors_lists = [cand_neighbors_list[_] for _ in I_sorted[0]]

    return new_cand_neighbors_lists, new_cand_neighbor_vs
# ====================== gnn  ===================

class CuiEmbedding(object):
    def __init__(self, embedding_dict, hdim, device):
        self.data = {
            cui: torch.tensor(emb_val, dtype=torch.float, device=device).squeeze()
            for cui, emb_val in embedding_dict.items()
        }
        self.embedding_dim = hdim
        self.device = device
    def encode(self, cui_str_list: list):
        return torch.stack([self.data.get(c, torch.zeros(self.embedding_dim, device=self.device)) for c in cui_str_list])
    def to(self, device):
        self.device = device
        self.data = {c: emb.to(device) for c, emb in self.data.items()}
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


class PathRanker(nn.Module):
    """
    Input: task embedding, cui embedding, and path embedding 
    Step 1: compute task relevancy and context relevancy 
    Step 2: compute attention scores based on task rel and context rel
    Module has been tested ok; Note that the return shape is B X 4*hdim 
    """
    def __init__(self, hdim, nums_of_head, attn_weight_mode="Linear", cui_flag=True):
        super(PathRanker, self).__init__()
        self.attention = nn.MultiheadAttention(4*hdim, nums_of_head)
        self.cui_flag = cui_flag
        self.attn_mode = attn_weight_mode
        self.mid_layer = nn.Linear(4*hdim, hdim)
        self.score = nn.Linear(hdim, 1)

        nn.init.xavier_uniform_(self.mid_layer.weight)
        nn.init.xavier_uniform_(self.score.weight)


    def forward(self, task_inputs, cui_inputs, path_embeddings):
        # Infersent based Task relevancy: input text (premise) and paths (hypothesis) 
        task_rel = torch.cat((task_inputs, 
                              path_embeddings, 
                              torch.abs(task_inputs - path_embeddings),
                          task_inputs * path_embeddings), 1)
        if self.cui_flag: # if also computing cui relevancy 
            cui_rel = torch.cat((cui_inputs, 
                                 path_embeddings, 
                                 torch.abs(cui_inputs - path_embeddings),
                          cui_inputs * path_embeddings), 1)
            #self.merge_repr = task_rel * cui_rel # Hadamard Product of the two matrices
            merge_repr = task_rel * cui_rel 
            attn_output, attn_output_weights = self.attention(merge_repr, merge_repr, merge_repr)
            self.attn_output_weights = attn_output_weights
        else:
            attn_output, attn_output_weights = self.attention(task_rel, task_rel, task_rel)

        scores = self.score(F.relu(self.mid_layer(attn_output)))

        return scores, attn_output, attn_output_weights # attn_output: weighted attention scores, B X 3072 ; attention output weights on scores 

# ====================== model  ===================

class GraphModel(nn.Module):
    def __init__(self,
                 g_nx, # NetworkX graph for preprocessing
                 cui_embedding_lookup, # CuiEmbedding object (lookup by CUI string)
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
                
        
        
        
        
        # Edge encoder needs the tensorized graph's edge_to_idx for dynamic mapping
        # Or, if EdgeOneHot is modified to take graph_nx and build its own mapping:
        self.e_encoder = EdgeOneHot(graph=g_nx, device=self.device) # Assumes EdgeOneHot can handle NX graph
        # self.edge_idx_map = self.g_tensorized['edge_to_idx'] # No longer needed if e_encoder handles it
        actual_edge_dim = self.e_encoder.num_edge_types
        
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
        self.path_per_batch_size = 256
        self.top_n = top_n
        self.cui_weights_dict = cui_weights_dict if cui_weights_dict else {}
        self.hdim = hdim # Store hdim for use
        
        self.gnn_update = gnn_update
        self.gnn_type = gnn_type
        self.gin_num_layers = gin_num_layers if gin_num_layers else (3 if gnn_type == "Stack" else 1)
        self.gin_hidden_dim = gin_hidden_dim if gin_hidden_dim else hdim
        #self.input_edge_dim_for_gin = input_edge_dim_for_gin # Should match e_encoder output
        if self.gnn_update:
            if self.gnn_type == "Stack":
                self.gnn = GINStack(
                    input_node_dim=hdim, 
                    input_edge_dim=actual_edge_dim, 
                    hidden_dim=self.gin_hidden_dim, 
                    output_dim_final=hdim, 
                    num_layers=self.gin_num_layers, 
                    device=device
                )
            else:
                self.gnn = NodeAggregateGIN(
                    input_node_dim=hdim, 
                    input_edge_dim=actual_edge_dim, 
                    hidden_dim=self.gin_hidden_dim, 
                    output_dim=hdim, 
                    device=device
                )
        else:
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
                      prev_iteration_state=None # Dict: {'cand_src_orig_idx': Tensor, 'cand_tgt_idx': Tensor}
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

        if cand_src_idx_hop.numel() == 0: # No paths found
            return None, {}, None, True # Scores, next_hop_dict, path_tensors, mem_tensors, stop_flag
        
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
            # print(f"Debug: Failed to get base embeddings for src or tgt at hop {running_k_hop}")
            return None, {}, None, True
        
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
        scatter_src_index_for_gin_agg = map_global_src_to_local_tensor[cand_src_idx_hop]

        # Get edge embeddings for all paths
        # EdgeOneHot.Lookup expects list of labels, convert cand_edge_idx_hop back to labels
        # --- 在 cand_edge_idx_hop 被定義之後 ---
        # 現在 EdgeOneHot.Lookup 直接接收索引張量
        # cand_edge_idx_hop 本身就是邊類型的索引 (0 到 num_edge_types-1)
        if cand_edge_idx_hop.numel() > 0: # 僅當有邊時才查找
            path_edge_embs_for_gin_and_path_enc = self.e_encoder.Lookup(cand_edge_idx_hop)
            # .to(self.device) 通常由 self.e_encoder.onehot_mat 初始化時或 Lookup 內部保證
        else: # 如果沒有邊索引 (例如，沒有路徑被找到)
            # actual_edge_dim 應在 __init__ 中從 self.e_encoder.num_edge_types 獲取
            actual_edge_dim = self.e_encoder.num_edge_types if self.e_encoder.num_edge_types > 0 else 0
            path_edge_embs_for_gin_and_path_enc = torch.empty(0, actual_edge_dim, device=self.device)


        current_path_src_embs_for_encoding = unique_src_embs_base.clone()
        if running_k_hop == 0 and self.cui_weights_dict:
            weights = torch.tensor(
                [self.cui_weights_dict.get(self.g_tensorized['idx_to_cui'].get(idx.item()), 1.0)
                 for idx in unique_hop_src_indices], device=self.device
            ).unsqueeze(1)
            current_path_src_embs_for_encoding = current_path_src_embs_for_encoding * weights
        
        
        # --- 3. Optional GNN Update on current_path_src_embs ---
        # This part is complex to tensorize fully without specific GIN assumptions.
        # We need to aggregate target+edge features for each source in unique_hop_src_indices.
        if self.gnn_update and self.gnn:

            # map_global_tgt_to_local_tensor 應該已經在前面根據 unique_hop_tgt_indices 創建好了
            # unique_tgt_embs 也已經準備好

            local_indices_for_gin_targets = map_global_tgt_to_local_tensor[cand_tgt_idx_hop]
            
            # 錯誤處理：檢查是否有 -1 索引
            if torch.any(local_indices_for_gin_targets == -1):
                print(f"警告 (gin_path_tgt_node_features): cand_tgt_idx_hop 中存在未映射到 unique_hop_tgt_indices 的索引。")
                # 根據您的策略處理錯誤，例如，如果決定過濾：
                # valid_mask = (local_indices_for_gin_targets != -1)
                # local_indices_for_gin_targets = local_indices_for_gin_targets[valid_mask]
                # # 其他與路徑對應的張量 (如 scatter_src_index_for_gin_agg, path_edge_embs_for_gin_and_path_enc)
                # # 也需要用這個 valid_mask 或等效方式進行篩選，以保持一致性。
                # # 這會使邏輯複雜化，所以更好的方式是確保 cand_tgt_idx_hop 中的所有索引都是有效的。
                # 暫時假設所有索引都有效，或者在發現-1時拋出錯誤以便調試：
                if torch.any(local_indices_for_gin_targets == -1):
                     raise ValueError("gin_path_tgt_node_features: 發現無效的-1索引，這表示 cand_tgt_idx_hop 與 unique_hop_tgt_indices 不一致。")

            # 確保 unique_tgt_embs 非空且索引有效
            if unique_tgt_embs.numel() > 0:
                gin_path_tgt_node_features = unique_tgt_embs[local_indices_for_gin_targets]
            elif cand_tgt_idx_hop.numel() > 0: # 有目標索引，但 unique_tgt_embs 為空 (不應發生)
                print(f"警告: unique_tgt_embs 為空，但 cand_tgt_idx_hop 並非如此。無法為 GNN 獲取有效的 gin_path_tgt_node_features。")
                gin_path_tgt_node_features = torch.empty(0, self.hdim, device=self.device)
            else: # cand_tgt_idx_hop 也為空
                gin_path_tgt_node_features = torch.empty(0, self.hdim, device=self.device)

            updated_src_embs_from_gin, _ = self.gnn(
                current_path_src_embs_for_encoding, # x_src_unique
                unique_hop_src_indices,             # unique_src_to_process_indices (global IDs)
                scatter_src_index_for_gin_agg,      # path_source_indices_global_scatter (local IDs for scatter)
                gin_path_tgt_node_features,         # path_target_node_features
                path_edge_embs_for_gin_and_path_enc # path_edge_features
            )
            current_path_src_embs_for_encoding = updated_src_embs_from_gin
            # print(f"GIN updated src embs shape: {current_path_src_embs_for_encoding.shape}")
            
        pruning_threshold_count = 256 # 您設定的篩選路徑數量上限
        num_paths_this_hop_before_pruning = num_paths_this_hop # ## NOW THIS IS VALID ##

        if num_paths_this_hop > pruning_threshold_count: # Check against initial num_paths_this_hop
            # print(f"  Hop {running_k_hop}: Path count {num_paths_this_hop} exceeds threshold {pruning_threshold_count}. Applying pruning.")

            if unique_tgt_embs.numel() > 0 : # 確保 unique_tgt_embs 不是空的
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
                    valid_first_edge_mask = (batch_mem_first_edge_idx != -1) & (batch_mem_first_edge_idx < self.e_encoder.onehot_mat.shape[0])
                    if torch.any(valid_first_edge_mask):
                        batch_projected_first_rel_embs = torch.zeros(batch_mem_first_edge_idx.size(0), self.hdim, device=self.device)
                        
                        first_rel_indices_to_lookup = batch_mem_first_edge_idx[valid_first_edge_mask]
                        if first_rel_indices_to_lookup.numel() > 0 and self.e_encoder.onehot_mat.numel() > 0 : # 確保 onehot_mat 非空
                            first_rel_one_hots = self.e_encoder.onehot_mat[first_rel_indices_to_lookup] # .to(self.device) 已在 onehot_mat 初始化時處理
                            batch_projected_first_rel_embs[valid_first_edge_mask] = self.edge_to_node_projection_for_transformer(first_rel_one_hots)
                
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
        return all_paths_info, visited_paths_str_dict, next_hop_state, stop_flag
    
class TestGraphModelOneIteration(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hdim = 9 # Small hidden dimension for tests
        
        # 1. Mock NetworkX Graph
        self.g_nx = nx.DiGraph()
        self.g_nx.add_edge("C001", "C002", label="RelA")
        self.g_nx.add_edge("C001", "C003", label="RelB")
        self.g_nx.add_edge("C002", "C004", label="RelC")
        self.g_nx.add_edge("C003", "C004", label="RelD")
        self.g_nx.add_node("C005") # Isolated node

        # 2. Mock CuiEmbedding object
        self.mock_cui_embeddings_dict = {
            "C001": np.random.rand(1, self.hdim).tolist(),
            "C002": np.random.rand(1, self.hdim).tolist(),
            "C003": np.random.rand(1, self.hdim).tolist(),
            "C004": np.random.rand(1, self.hdim).tolist(),
            "C005": np.random.rand(1, self.hdim).tolist(),
        }
        self.cui_embedding_lookup = CuiEmbedding(self.mock_cui_embeddings_dict, self.hdim, self.device)

        # 3. Mock cui_weights_dict
        self.cui_weights = {"C001": 1.5, "C002": 0.8}

        # Task and Context Embeddings (Batch size of 1 for simplicity in testing one_iteration)
        self.task_emb = torch.randn(1, self.hdim, device=self.device)
        self.context_emb = torch.randn(1, self.hdim, device=self.device)
        
        # Global cui_to_idx and edge_to_idx will be set by preprocess_graph_to_tensors
        # This happens inside GraphModel.__init__

    def _init_graph_model(self, gnn_update=False, path_encoder_type="MLP", gnn_type="Stack", top_n=3, num_hops=2):
        model = GraphModel(
            g_nx=self.g_nx,
            cui_embedding_lookup=self.cui_embedding_lookup,
            hdim=self.hdim,
            nums_of_head=2, # Example
            num_hops=num_hops,
            top_n=top_n,
            device=self.device,
            cui_weights_dict=self.cui_weights,
            gnn_update=gnn_update,
            path_encoder_type=path_encoder_type,
            gnn_type=gnn_type,
            gin_num_layers=1 # Simplified for testing GINStack
        )
        model.to(self.device)
        return model

    def test_one_iteration_1hop_no_paths(self):
        """Test case where starting CUI has no outgoing paths."""
        graph_model = self._init_graph_model()
        start_cuis = ["C005"] # Isolated node
        
        all_paths_info, _, next_state, stop_flag = graph_model.one_iteration(
            self.task_emb, start_cuis, running_k_hop=0, context_emb_batch=self.context_emb
        )
        self.assertIsNone(all_paths_info) # Expecting None when no paths
        self.assertTrue(stop_flag)

    def test_one_iteration_1hop_basic_mlp(self):
        """Test 1-hop inference with MLP PathEncoder."""
        top_n_test = 2
        graph_model = self._init_graph_model(gnn_update=False, path_encoder_type="MLP", top_n=top_n_test)
        start_cuis = ["C001"]
        
        all_paths_info, _, next_state, stop_flag = graph_model.one_iteration(
            self.task_emb, start_cuis, running_k_hop=0, context_emb_batch=self.context_emb
        )
        self.assertFalse(stop_flag)
        self.assertIsNotNone(all_paths_info)
        
        # C001 has 2 outgoing paths: C001->RelA->C002, C001->RelB->C003
        self.assertEqual(all_paths_info["scores"].shape[0], 2) # Number of paths from C001
        self.assertEqual(all_paths_info["encoded_embeddings"].shape, (2, self.hdim))
        self.assertEqual(all_paths_info["tgt_idx"].numel(), 2)

        self.assertIsNotNone(next_state)
        self.assertEqual(next_state["selected_hop_target_idx"].numel(), top_n_test) # top_n paths selected
        self.assertIsNotNone(next_state["selected_src_orig_idx"])
        self.assertEqual(next_state["selected_src_orig_idx"].numel(), top_n_test)
        # For 1st hop, selected_first_hop_edge_idx should be the edges of the current hop
        self.assertIsNotNone(next_state["selected_first_hop_edge_idx"])
        self.assertEqual(next_state["selected_first_hop_edge_idx"].numel(), top_n_test)
        
        # Check if original source is C001 for selected paths
        c001_idx = graph_model.g_tensorized['cui_to_idx']["C001"]
        self.assertTrue(torch.all(next_state["selected_src_orig_idx"] == c001_idx))


    def test_one_iteration_1hop_cui_weights_applied(self):
        graph_model = self._init_graph_model(gnn_update=False, path_encoder_type="MLP")
        start_cuis = ["C001"] # C001 has weight 1.5
        
        # To check weight application, we'd ideally need to inspect 
        # `current_path_src_embs_for_encoding` inside `one_iteration`
        # or mock `_get_embeddings_by_indices` and `p_encoder` to verify inputs.
        # This is more of an integration check.
        # We can check if the scores are produced.
        with patch.object(graph_model, '_get_embeddings_by_indices', wraps=graph_model._get_embeddings_by_indices) as mock_get_emb:
            all_paths_info, _, _, _ = graph_model.one_iteration(
                self.task_emb, start_cuis, running_k_hop=0, context_emb_batch=self.context_emb
            )
            self.assertIsNotNone(all_paths_info)
            # Difficult to assert the exact numerical effect of weighting without deeper mocking
            # or specific values that would make the effect obvious.
            # For now, confirm it runs. In a more detailed test, one could mock `_get_embeddings_by_indices`
            # to return known values and then check if `current_path_src_embs_for_encoding` reflects the weight.


    def test_one_iteration_2hop_mlp_memory_passing(self):
        """Test 2-hop, focusing on how memory (orig_src, first_edge) is passed."""
        top_n_test = 1 # Select one path from 1st hop to make 2nd hop predictable
        graph_model = self._init_graph_model(gnn_update=False, path_encoder_type="MLP", top_n=top_n_test)
        
        # --- Mock 1st Hop ---
        start_cuis_hop1 = ["C001"]
        # Manually create a plausible prev_iteration_state for 2nd hop
        # Say, 1st hop selected C001 -> RelA -> C002
        c001_idx = graph_model.g_tensorized['cui_to_idx']["C001"]
        c002_idx = graph_model.g_tensorized['cui_to_idx']["C002"]
        rela_idx = graph_model.g_tensorized['edge_to_idx']["RelA"]

        prev_state_for_hop2 = {
            "selected_src_orig_idx": torch.tensor([c001_idx], device=self.device),
            "selected_hop_target_idx": torch.tensor([c002_idx], device=self.device), # This becomes start CUI for hop 2
            "selected_first_hop_edge_idx": torch.tensor([rela_idx], device=self.device)
        }
        start_cuis_hop2 = ["C002"] # Target from hop1

        # --- Execute 2nd Hop ---
        all_paths_info_hop2, _, next_state_hop2, stop_flag_hop2 = graph_model.one_iteration(
            self.task_emb, start_cuis_hop2, running_k_hop=1, 
            context_emb_batch=self.context_emb, prev_iteration_state=prev_state_for_hop2
        )
        self.assertFalse(stop_flag_hop2)
        self.assertIsNotNone(all_paths_info_hop2)
        # C002 has 1 outgoing path: C002 -> RelC -> C004
        self.assertEqual(all_paths_info_hop2["scores"].shape[0], 1) 
        
        self.assertIsNotNone(next_state_hop2)
        self.assertEqual(next_state_hop2["selected_hop_target_idx"].numel(), top_n_test) # Should be C004
        c004_idx = graph_model.g_tensorized['cui_to_idx']["C004"]
        self.assertEqual(next_state_hop2["selected_hop_target_idx"][0].item(), c004_idx)

        # Crucial checks for memory:
        # Original source should still be C001
        self.assertEqual(next_state_hop2["selected_src_orig_idx"][0].item(), c001_idx)
        # First hop edge should still be RelA
        self.assertEqual(next_state_hop2["selected_first_hop_edge_idx"][0].item(), rela_idx)

    def test_one_iteration_path_pruning(self):
        graph_model_for_prune = self._init_graph_model(top_n=2)
        start_cuis = ["C001"] 

        # 修改這裡：wraps 實例上的方法
        with patch.object(graph_model_for_prune, 'one_iteration', wraps=graph_model_for_prune.one_iteration) as wrapped_one_iter_on_instance:
            all_paths_info, _, _, _ = graph_model_for_prune.one_iteration(
                task_emb_batch=self.task_emb, 
                current_cui_str_list=start_cuis, 
                running_k_hop=0,
                context_emb_batch=self.context_emb, # 確保所有需要的參數都被傳遞
                prev_iteration_state=None
            )
            
            wrapped_one_iter_on_instance.assert_called_once_with(
                task_emb_batch=self.task_emb,
                current_cui_str_list=start_cuis,
                running_k_hop=0,
                context_emb_batch=self.context_emb,
                prev_iteration_state=None
            )
            
            if all_paths_info is not None and "scores" in all_paths_info:
                 self.assertEqual(all_paths_info["scores"].shape[0], 2) 
            else:
                 self.fail("one_iteration did not return expected path_info dictionary or was None")

    def test_one_iteration_gnn_update_flow(self):
        graph_model_gnn = self._init_graph_model(gnn_update=True, gnn_type="Stack", top_n=1)
        start_cuis = ["C001"]

        # Mock the GNN module itself to verify it's called and its output is used.
        mock_gnn_output_emb = torch.randn(1, self.hdim, device=self.device) # GNN updates unique_src_embs
        
        # Since GNN is part of graph_model, we patch its forward method
        with patch.object(graph_model_gnn.gnn, 'forward', return_value=(mock_gnn_output_emb, torch.tensor([graph_model_gnn.g_tensorized['cui_to_idx']['C001']], device=self.device))) as mock_gnn_forward:
            with patch.object(graph_model_gnn.p_encoder, 'forward') as mock_penc_forward:
                # Mock path encoder to see what src embeddings it receives
                mock_penc_forward.return_value = torch.randn(graph_model_gnn.g_tensorized['adj_src'].numel(), self.hdim, device=self.device)[:2,:] # Mock output for 2 paths from C001

                graph_model_gnn.one_iteration(
                    self.task_emb, start_cuis, running_k_hop=0
                )
                mock_gnn_forward.assert_called_once()
                
                # Check if p_encoder received GNN's output (indirectly)
                # The first argument to p_encoder.forward is src_b (path_specific_src_embs)
                # which should be derived from current_path_src_embs_for_encoding, which GNN updated.
                # This is tricky to assert directly without more complex mocking of intermediate states.
                # For now, asserting GNN was called is a good first step.
                # And that p_encoder was subsequently called.
                mock_penc_forward.assert_called()


    def test_one_iteration_transformer_path_encoder_1hop(self):
        graph_model_transformer = self._init_graph_model(path_encoder_type="Transformer", top_n=1)
        start_cuis = ["C001"] # C001 -> C002 (RelA), C001 -> C003 (RelB)

        with patch.object(graph_model_transformer.p_encoder, 'forward') as mock_transformer_forward:
            # Mock return value to be [batch_path_encoder, hdim]
            # If 2 paths are processed in the batch for path_encoder
            mock_transformer_forward.return_value = torch.randn(2, self.hdim, device=self.device) 

            graph_model_transformer.one_iteration(self.task_emb, start_cuis, running_k_hop=0)
            
            mock_transformer_forward.assert_called()
            # Args to PathEncoderTransformer: forward(self, src_seq, tgt_edge_combined_single_step)
            # For 1-hop, src_seq should ideally be based on just the source node embedding
            # e.g., shape [num_paths_in_batch, 1, hdim] if seq_len is 1
            call_args = mock_transformer_forward.call_args[0]
            src_seq_arg = call_args[0] # src_b_for_pencoder
            
            # print(f"Transformer 1-hop src_seq_arg shape: {src_seq_arg.shape}")
            # Expected sequence for 1-hop: [Orig_Src_Emb]
            # So, seq_len should be 1 if correctly implemented.
            self.assertEqual(src_seq_arg.ndim, 3) # [batch_size_for_penc, seq_len, hdim]
            # Based on your Transformer PEncoder input logic:
            # For hop 0: path_elements_embs_list will contain [current_src_emb]
            # So seq_len after pad_sequence will be 1.
            if src_seq_arg.numel() > 0: # If any paths were processed
                 self.assertEqual(src_seq_arg.shape[1], 1, "Transformer src_seq for 1-hop should have seq_len 1")

    @patch('__main__.retrieve_neighbors_paths_no_self_tensorized') # Patch at the point of import for the test file
    def test_one_iteration_transformer_path_encoder_2hop(self, mock_retrieve_paths):
        graph_model_transformer = self._init_graph_model(path_encoder_type="Transformer", top_n=1)
        
        # Mocking the output of retrieve_neighbors_paths_no_self_tensorized for a 2-hop scenario
        # Say current CUI is C002, original was C001 via RelA
        c001_idx = graph_model_transformer.g_tensorized['cui_to_idx']["C001"]
        c002_idx = graph_model_transformer.g_tensorized['cui_to_idx']["C002"]
        c004_idx = graph_model_transformer.g_tensorized['cui_to_idx']["C004"]
        rela_idx = graph_model_transformer.g_tensorized['edge_to_idx']["RelA"]
        relc_idx = graph_model_transformer.g_tensorized['edge_to_idx']["RelC"]

        mock_retrieve_paths.return_value = (
            torch.tensor([c002_idx], device=self.device),                 # cand_src_idx_hop (current src: C002)
            torch.tensor([c004_idx], device=self.device),                 # cand_tgt_idx_hop (target: C004)
            torch.tensor([relc_idx], device=self.device),                 # cand_edge_idx_hop (edge: RelC)
            torch.tensor([c001_idx], device=self.device),                 # mem_orig_src_idx_hop (original src: C001)
            torch.tensor([rela_idx], device=self.device)                  # mem_first_edge_idx_hop (first edge: RelA)
        )
        
        start_cuis_hop2 = ["C002"]
        # This prev_state isn't directly used by the patched retrieve_paths, but good for context
        prev_state_for_hop2 = { 
            "selected_src_orig_idx": torch.tensor([c001_idx], device=self.device),
            "selected_hop_target_idx": torch.tensor([c002_idx], device=self.device),
            "selected_first_hop_edge_idx": torch.tensor([rela_idx], device=self.device)
        }

        with patch.object(graph_model_transformer.p_encoder, 'forward') as mock_transformer_forward:
            mock_transformer_forward.return_value = torch.randn(1, self.hdim, device=self.device) # 1 path in this mock

            graph_model_transformer.one_iteration(
                self.task_emb, start_cuis_hop2, running_k_hop=1, prev_iteration_state=prev_state_for_hop2
            )
            
            mock_transformer_forward.assert_called_once()
            call_args = mock_transformer_forward.call_args[0]
            src_seq_arg = call_args[0] # src_b_for_pencoder
            
            # Expected sequence for 2-hop: [Orig_Src_Emb, First_Rel_Emb, Intermediate_Node_Emb (current_src)]
            # So, seq_len should be 3 if correctly implemented.
            # print(f"Transformer 2-hop src_seq_arg shape: {src_seq_arg.shape}")
            if src_seq_arg.numel() > 0:
                 self.assertEqual(src_seq_arg.ndim, 3)
                 self.assertEqual(src_seq_arg.shape[1], 3, "Transformer src_seq for 2-hop should have seq_len 3")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
