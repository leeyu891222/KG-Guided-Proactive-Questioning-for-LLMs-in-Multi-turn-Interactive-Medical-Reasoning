import unittest
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
    from torch_scatter import scatter_add, scatter_mean
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
        self.valid_training_samples = [] # 將存儲 (case_id, known_idx, unknown_idx)

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
        num_valid_pairs_found = 0

        for case_id, data in all_annotations.items():
            atomic_facts = data.get("atomic_facts", [])
            facts_cuis = data.get("facts_cuis", [])
            paths_between_facts = data.get("paths_between_facts", {})
            num_facts = len(atomic_facts)

            if num_facts < self.min_facts_for_sampling or len(facts_cuis) != num_facts:
                continue
            num_cases_after_min_facts_filter += 1

            # 找出所有存在預計算路徑的 (known_fact_idx, unknown_fact_idx) 對
            for i in range(num_facts):
                for j in range(num_facts):
                    if i == j:  # 已知和未知不能是同一個 fact
                        continue
                    
                    path_key = f"{i}_{j}" # 假設 i 是 known, j 是 unknown
                    if path_key in paths_between_facts and paths_between_facts[path_key]:
                        # 確保 unknown_fact_idx 对应的 CUI 列表不为空
                        if facts_cuis[j]: # 如果 unknown fact 有 CUI
                            # 进一步确保路径的目标CUI至少有一个在unknown fact的CUI列表中
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
                                    # "all_case_data": data # 可以選擇是否存儲整個案例數據以避免重複查找
                                })
                                num_valid_pairs_found +=1
        
        self.all_annotations_data = all_annotations # 存儲所有數據，供後續查找

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
        
        # 從 self.all_annotations_data 中獲取該 case 的完整數據
        data = self.all_annotations_data[case_id]
        atomic_facts = data["atomic_facts"]
        facts_cuis = data["facts_cuis"]
        paths_between_facts = data.get("paths_between_facts", {})
        num_facts = len(atomic_facts)

        # --- 構建已知集和未知集 ---
        known_indices = {guaranteed_known_idx}
        unknown_indices = {guaranteed_unknown_idx} # 初始未知集只包含這個保證的未知

        # 可選：從剩餘 facts 中隨機添加更多 facts 到已知集或未知集，以增加多樣性
        remaining_indices = list(set(range(num_facts)) - known_indices - unknown_indices)
        self.random.shuffle(remaining_indices)

        # 例如，將剩餘的一半（隨機）加入已知集，另一半加入未知集（如果需要更多未知目標）
        # 或者更簡單：除了 guaranteed_unknown_idx 之外，其他都作為已知（如果適用）
        # 為了保證至少有一個確定的 GT 來源，我們先這樣設計：
        # known_indices 可以擴展，但 unknown_indices 的 GT 主要來自 guaranteed_unknown_idx

        # 示例：將一部分 remaining_indices 加入 known_indices
        if remaining_indices:
            num_additional_known = self.random.randint(0, len(remaining_indices))
            known_indices.update(remaining_indices[:num_additional_known])
            # 剩下的可以加入 unknown_indices，或者作為模型的 "distractor" / "context"
            # 如果我們只關心從 known_indices 到 guaranteed_unknown_idx 的路徑，則其他 unknown 可以忽略
            # 為了與方案一的目標一致（找到從已知到任何未知的路徑），我們應該將剩餘的也加入 unknown_indices
            unknown_indices.update(remaining_indices[num_additional_known:])


        # --- 準備模型輸入 (Input Text 和 Known CUIs) ---
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

        # --- 從 guaranteed_known_idx 到 guaranteed_unknown_idx 提取 GT CUIs ---
        # 由於我們是從確保有路徑的 (guaranteed_known_idx, guaranteed_unknown_idx) 開始的，
        # 所以這裡 paths_for_this_guaranteed_pair 幾乎一定會有內容。
        hop1_target_cuis_set = set()
        hop2_target_cuis_set = set()
        
        # 我們也可以考慮從所有 known_indices 到所有 unknown_indices 的路徑
        # 但為了確保 __getitem__ 總能基於 "guaranteed" pair 產生 GT，我們先專注於這個核心連接
        # 然後可以選擇性地加入其他 known 到其他 unknown 的 GT
        
        primary_path_key = f"{guaranteed_known_idx}_{guaranteed_unknown_idx}"
        paths_for_this_guaranteed_pair = paths_between_facts.get(primary_path_key, [])

        if not paths_for_this_guaranteed_pair:
            # 這種情況理論上不應該發生，因為 self.valid_training_samples 的構建已經篩選了
            # 但作為防禦性編程，如果發生了，返回 None
            print(f"嚴重警告: 案例 {case_id}, 預期有路徑的 guaranteed_pair ({guaranteed_known_idx}_{guaranteed_unknown_idx}) 未找到路徑！")
            return None

        for path_data in paths_for_this_guaranteed_pair:
            if not path_data or not isinstance(path_data, list): continue
            path_len = len(path_data)
            target_cui_in_path = path_data[-1]
            is_valid_cui = isinstance(target_cui_in_path, str) and target_cui_in_path.startswith('C')

            if is_valid_cui and isinstance(facts_cuis[guaranteed_unknown_idx], list) and \
               target_cui_in_path in facts_cuis[guaranteed_unknown_idx]:
                if path_len == 3:
                    hop1_target_cuis_set.add(target_cui_in_path)
                    hop2_target_cuis_set.add(target_cui_in_path)
                elif path_len == 5:
                    hop2_target_cuis_set.add(target_cui_in_path)

        # 可選：遍歷所有 known_indices 到所有 (除了 guaranteed_unknown_idx 之外的) unknown_indices
        # 以增加更多樣的 GT，但要確保這不會導致 GT 經常為空
        for k_idx in known_indices:
            for u_idx in unknown_indices:
                if u_idx == guaranteed_unknown_idx and k_idx == guaranteed_known_idx: # 已經處理過 guaranteed pair
                    continue
                path_key_optional = f"{k_idx}_{u_idx}"
                if path_key_optional in paths_between_facts:
                    for path_data in paths_between_facts[path_key_optional]:
                        if not path_data or not isinstance(path_data, list): continue
                        path_len = len(path_data)
                        target_cui_in_path = path_data[-1]
                        is_valid_cui = isinstance(target_cui_in_path, str) and target_cui_in_path.startswith('C')
                        if is_valid_cui and isinstance(facts_cuis[u_idx], list) and \
                           target_cui_in_path in facts_cuis[u_idx]:
                            if path_len == 3:
                                hop1_target_cuis_set.add(target_cui_in_path) # 加入到set中自然去重
                                hop2_target_cuis_set.add(target_cui_in_path)
                            elif path_len == 5:
                                hop2_target_cuis_set.add(target_cui_in_path)
        
        # 最終檢查，如果經過所有努力，hop2 依然是空的（理論上不應發生，因為 guaranteed pair）
        if not hop2_target_cuis_set:
            # print(f"警告: 案例 {case_id} (索引 {index}) 即使使用 guaranteed_pair，hop2_target_cuis 仍為空。")
            return None # 這種情況值得調查

        return {
            "case_id": case_id,
            "known_indices": sorted(list(known_indices)),
            "unknown_indices": sorted(list(unknown_indices)),
            "input_text_tks": input_text_tks,
            "known_cuis": known_cuis_list_unique,
            "hop1_target_cuis": list(hop1_target_cuis_set),
            "hop2_target_cuis": list(hop2_target_cuis_set)
        }


# --- 修改後的 Collate Function ---
def collate_fn_mediq_paths(batch):
    valid_items_in_batch = [item for item in batch if item is not None]
    if not valid_items_in_batch:
        return None
    case_ids = [item['case_id'] for item in valid_items_in_batch]
    # known_indices = [item['known_indices'] for item in valid_items_in_batch]
    # unknown_indices = [item['unknown_indices'] for item in valid_items_in_batch]
    input_ids = [item['input_text_tks']['input_ids'] for item in valid_items_in_batch]
    attention_mask = [item['input_text_tks']['attention_mask'] for item in valid_items_in_batch]
    known_cuis = [item['known_cuis'] for item in valid_items_in_batch]
    hop1_target_cuis = [item['hop1_target_cuis'] for item in valid_items_in_batch]
    hop2_target_cuis = [item['hop2_target_cuis'] for item in valid_items_in_batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        "case_id": case_ids,
        # "known_indices": known_indices,
        # "unknown_indices": unknown_indices,
        "input_text_tks_padded": {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded
        },
        "known_cuis": known_cuis,
        "hop1_target_cuis": hop1_target_cuis,
        "hop2_target_cuis": hop2_target_cuis
    }
    


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
    """
    Loads pre-computed CUI embeddings from a pickle file.
    Handles values that are numpy.ndarray of shape (1, embedding_dim).
    """
    def __init__(self, embedding_file_path, device=torch.device('cpu')):
        print(f"載入真實 CUI 嵌入從: {embedding_file_path}")
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

class MockCuiEmbedding:
    def __init__(self, data, hdim, device): self.data = data; self.hdim = hdim; self.device = device
    def encode(self, cuis_str): 
        return torch.stack([self.data.get(c, torch.randn(1,self.hdim).squeeze(0)) for c in cuis_str]).to(self.device)

class EdgeOneHot(object): # Using the dynamic one from user's code
    def __init__(self, graph: nx.DiGraph, unknown_rel_label: str = "UNKNOWN_REL"):
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
        if self.num_edge_types == 0: # Handle graph with no edges or no labeled edges
            # print("Warning: EdgeOneHot found 0 edge types. onehot_mat will be empty or minimal.")
            # This might lead to issues if edge_dim is expected to be >0 later
            # For now, create a minimal valid onehot_mat if num_edge_types would be 0
            self.onehot_mat = torch.empty(0,0).float() # Or torch.zeros(1,1).float() if a min dim is needed
        else:
            self.onehot_mat = F.one_hot(torch.arange(0, self.num_edge_types), num_classes=self.num_edge_types).float()
    def Lookup(self, edge_labels: list):
        if self.num_edge_types == 0: # No edge types defined
            # Return a zero tensor of expected dimensionality if possible, or empty.
            # This depends on what PathEncoder expects as edge_feature_dim_for_path_encoder.
            # If EdgeOneHot created a (0,0) mat, this will error.
            # Let's assume if num_edge_types is 0, edge_dim is 0.
            # PathEncoder should handle edge_dim=0 (e.g., not concat edge embs).
            if hasattr(self, '_expected_edge_dim_downstream') and self._expected_edge_dim_downstream > 0:
                 return torch.zeros(len(edge_labels), self._expected_edge_dim_downstream) # Fallback
            return torch.empty(len(edge_labels), 0) # Correct if edge_dim is 0

        indices = []
        for e in edge_labels:
            if e in self.edge_mappings: indices.append(self.edge_mappings[e])
            elif self.unknown_rel_index is not None: indices.append(self.unknown_rel_index)
            else: indices.append(0) # Fallback
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        indices_tensor = torch.clamp(indices_tensor, 0, self.num_edge_types - 1 if self.num_edge_types > 0 else 0)
        if self.num_edge_types == 0 : # Should not happen if clamp works correctly
            return torch.empty(len(edge_labels), 0)
        return self.onehot_mat[indices_tensor]

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
            # Efficient aggregation using torch_scatter
            # 1. Combine target node features and edge features for all paths
            combined_neighbor_edge_feats = torch.cat((path_target_node_features, path_edge_features), dim=-1)
            
            # 2. Transform these combined features (message from neighbor+edge)
            # Output shape: [total_paths_from_these_srcs, hidden_dim]
            transformed_messages = F.relu(self.edge_transform_linear(combined_neighbor_edge_feats))
            
            # 3. Aggregate messages for each source node
            # We need to map global `path_source_indices_global` to local indices [0, num_unique_src-1]
            # This mapping should align with `x_src_unique`.
            # `unique_src_to_process_indices` gives the global indices of nodes in `x_src_unique`.
            # We need to find where each `path_source_indices_global` entry falls in `unique_src_to_process_indices`.
            
            # A simple way if unique_src_to_process_indices is sorted and path_source_indices_global contains its elements:
            # (This assumes path_source_indices_global are already mapped or can be mapped to 0..N-1 indices for scatter)
            # For scatter, the `index` argument should be local indices for the output tensor.
            
            # Let's create the local scatter_index:
            # map_global_to_local_idx = {global_idx.item(): i for i, global_idx in enumerate(unique_src_to_process_indices)}
            # scatter_index = torch.tensor(
            #     [map_global_to_local_idx[glob_idx.item()] for glob_idx in path_source_indices_global],
            #     dtype=torch.long, device=self.device
            # )
            # --- More robust way to get scatter_index ---
            # Create a "compressed" version of path_source_indices_global
            # Example: unique_src_to_process_indices = [10, 20, 30] (global IDs)
            #          path_source_indices_global = [10, 20, 10, 30, 20] (global IDs)
            # We want scatter_index = [0, 1, 0, 2, 1] (local IDs relative to unique_src_to_process_indices)
            # This can be achieved using searchsorted if unique_src_to_process_indices is sorted.
            # Or by building a mapping.

            # Assuming unique_src_to_process_indices is sorted for searchsorted to work efficiently
            # If not, sort it and remember the inverse permutation if original order matters for x_src_unique.
            # For simplicity, let's assume unique_src_to_process_indices directly corresponds to the 0..N-1 order of x_src_unique.
            # This means path_source_indices_global should already be local indices [0, num_unique_src-1]
            # for scatter_add's `index` argument.
            # If path_source_indices_global are still global, they need to be mapped.

            # Let's refine this: `GraphModel.one_iteration` should prepare a `scatter_src_index`
            # that directly maps paths to the 0..N-1 indices of `x_src_unique`.
            # For this example, let's assume `path_source_indices_global` IS this scatter_index.
            # (This means GraphModel needs to prepare it correctly based on map_src_idx_to_unique_emb_idx)

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
        
        # Edge encoder needs the tensorized graph's edge_to_idx for dynamic mapping
        # Or, if EdgeOneHot is modified to take graph_nx and build its own mapping:
        self.e_encoder = EdgeOneHot(graph=g_nx) # Assumes EdgeOneHot can handle NX graph
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
        self.path_per_batch_size = 99999
        self.top_n = top_n
        self.cui_weights_dict = cui_weights_dict if cui_weights_dict else {}
        self.hdim = hdim # Store hdim for use
        self.device = device
        self.gnn_update = gnn_update
        self.gnn_type = gnn_type
        self.gin_num_layers = gin_num_layers if gin_num_layers else (3 if gnn_type == "Stack" else 1)
        self.gin_hidden_dim = gin_hidden_dim if gin_hidden_dim else hdim
        #self.input_edge_dim_for_gin = input_edge_dim_for_gin # Should match e_encoder output
        if self.gnn_update:
            if self.gnn_type == "Stack":
                self.gnn = GINStack(
                    input_node_dim=hdim, 
                    input_edge_dim=actual_edge_dim, # 使用真實值
                    hidden_dim=self.gin_hidden_dim, 
                    output_dim_final=hdim, 
                    num_layers=self.gin_num_layers, 
                    device=device
                )
            else:
                self.gnn = NodeAggregateGIN(
                    input_node_dim=hdim, 
                    input_edge_dim=actual_edge_dim, # 使用真實值
                    hidden_dim=self.gin_hidden_dim, 
                    output_dim=hdim, 
                    device=device
                )
        else:
            self.gnn = None
        
        

    def _get_embeddings_by_indices(self, cui_indices_tensor):
        """Helper to get embeddings for a tensor of CUI indices."""
        if cui_indices_tensor is None or cui_indices_tensor.numel() == 0:
            return torch.tensor([], device=self.device) # Return empty tensor on device
        
        # Convert indices back to CUI strings for lookup (inefficient, but uses existing n_encoder_lookup)
        # Ideal: n_encoder_lookup.encode_by_idx(cui_indices_tensor)
        idx_to_cui_map = self.g_tensorized['idx_to_cui']
        cui_strings = [idx_to_cui_map.get(idx.item()) for idx in cui_indices_tensor]
        
        # Filter out None if any index was not in map (should not happen if indices are from graph)
        valid_cui_strings = [s for s in cui_strings if s is not None]
        if not valid_cui_strings:
            return torch.tensor([], device=self.device)
            
        # This will return a tensor of shape [num_valid_cuis, 1, hdim] if encode expects list
        # or [num_valid_cuis, hdim] if it handles batching.
        # CuiEmbedding.encode returns [N, 1, D], so squeeze later.
        try:
            embeddings = self.n_encoder_lookup.encode(valid_cui_strings).to(self.device) # Ensure on device
            # We need to map these back to the original order of cui_indices_tensor if some were invalid
            # For now, assume all indices map to valid CUIs and are found by encode
            if len(valid_cui_strings) != cui_indices_tensor.numel():
                 print(f"Warning: Some CUI indices could not be mapped or embedded. Original: {cui_indices_tensor.numel()}, Valid: {len(valid_cui_strings)}")
                 # This part is tricky: how to create a tensor of correct size with zeros for missing?
                 # For simplicity, we proceed with only valid embeddings. This might break downstream if sizes don't match.
                 # A robust solution would involve creating a zero tensor and filling it.
            return embeddings.squeeze(1) # Shape [N, D]
        except KeyError as e:
            print(f"KeyError during embedding lookup in _get_embeddings_by_indices: {e}")
            return torch.tensor([], device=self.device)
        except Exception as e:
            print(f"Unexpected error in _get_embeddings_by_indices: {e}")
            return torch.tensor([], device=self.device)


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
            
        map_global_src_idx_to_local_in_unique = {
            glob_idx.item(): i for i, glob_idx in enumerate(unique_hop_src_indices)
        }
        # This scatter_index maps each path's global source CUI to its 0..N-1 index within unique_hop_src_indices
        scatter_src_index_for_gin_agg = torch.tensor(
            [map_global_src_idx_to_local_in_unique[glob_idx.item()] for glob_idx in cand_src_idx_hop],
            dtype=torch.long, device=self.device
        )

        # Get edge embeddings for all paths
        # EdgeOneHot.Lookup expects list of labels, convert cand_edge_idx_hop back to labels
        idx_to_edge_map = self.g_tensorized['idx_to_edge']
        edge_labels_for_paths = [idx_to_edge_map.get(idx.item(), "UNKNOWN_REL_LOOKUP") for idx in cand_edge_idx_hop]
        path_edge_embs_for_gin_and_path_enc = self.e_encoder.Lookup(edge_labels_for_paths).to(self.device)


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

            map_global_tgt_idx_to_local_in_unique = {
                 glob_idx.item(): i for i, glob_idx in enumerate(unique_hop_tgt_indices)
            }
            gin_path_tgt_node_features = unique_tgt_embs[
                torch.tensor([map_global_tgt_idx_to_local_in_unique[glob_idx.item()] for glob_idx in cand_tgt_idx_hop],
                             dtype=torch.long, device=self.device)
            ]

            updated_src_embs_from_gin, _ = self.gnn(
                current_path_src_embs_for_encoding, # x_src_unique
                unique_hop_src_indices,             # unique_src_to_process_indices (global IDs)
                scatter_src_index_for_gin_agg,      # path_source_indices_global_scatter (local IDs for scatter)
                gin_path_tgt_node_features,         # path_target_node_features
                path_edge_embs_for_gin_and_path_enc # path_edge_features
            )
            current_path_src_embs_for_encoding = updated_src_embs_from_gin
            # print(f"GIN updated src embs shape: {current_path_src_embs_for_encoding.shape}")

        # --- 5. Path Encoding and Ranking (Mini-batch loop) ---
        num_paths_this_hop = cand_src_idx_hop.size(0)
        all_path_scores_hop, all_encoded_paths_hop = [], []

        for i in range(0, num_paths_this_hop, self.path_per_batch_size):
            s_ = slice(i, min(i + self.path_per_batch_size, num_paths_this_hop))

            if self.p_encoder_type == "Transformer":
                src_sequences_for_transformer = []
                # 映射：全局索引 -> 在 unique 張量中的局部索引
                map_mem_orig_to_local = {glob_idx.item(): i for i, glob_idx in enumerate(unique_mem_orig_src_indices)} if unique_mem_orig_src_indices is not None else {}
                map_curr_src_to_local = map_global_src_idx_to_local_in_unique

                for j in range(s_.start, s_.stop): # 遍歷當前小批次中的每一條路徑
                    path_elements_embs = []
                    
                    # 1. 添加最初始源節點嵌入 (CUI_A)
                    orig_src_idx_val = mem_orig_src_idx_hop[j].item() if mem_orig_src_idx_hop is not None else -1
                    if orig_src_idx_val != -1 and orig_src_idx_val in map_mem_orig_to_local and unique_mem_orig_src_embs is not None:
                        path_elements_embs.append(unique_mem_orig_src_embs[map_mem_orig_to_local[orig_src_idx_val]])
                    else: # 如果是第一跳，orig_src 就是 curr_src
                        curr_src_idx_val = cand_src_idx_hop[j].item()
                        if curr_src_idx_val in map_curr_src_to_local : # 確保 curr_src_idx_val 在映射中
                             path_elements_embs.append(current_path_src_embs_for_encoding[map_curr_src_to_local[curr_src_idx_val]])


                    # 2. 添加第一個關係的嵌入 (Rel1) - 僅當是第二跳且 Rel1 存在時
                    if running_k_hop == 1 and mem_first_edge_idx_hop is not None:
                        first_edge_idx_val = mem_first_edge_idx_hop[j].item()
                        if first_edge_idx_val != -1: # -1 是無效/未知填充值
                            if first_edge_idx_val < self.e_encoder.onehot_mat.shape[0]: # 邊界檢查
                                first_rel_one_hot = self.e_encoder.onehot_mat[first_edge_idx_val].to(self.device)
                                projected_first_rel_emb = self.edge_to_node_projection_for_transformer(first_rel_one_hot)
                                path_elements_embs.append(projected_first_rel_emb)
                            else:
                                print(f"Warning: first_edge_idx_val {first_edge_idx_val} out of bounds for e_encoder.onehot_mat shape {self.e_encoder.onehot_mat.shape}")
                                # 可以選擇添加一個零向量或採取其他錯誤處理
                                path_elements_embs.append(torch.zeros(self.hdim, device=self.device))


                    # 3. 添加中間節點嵌入 (CUI_B) - 僅當是第二跳且與最初始源節點不同時
                    curr_src_idx_val = cand_src_idx_hop[j].item()
                    if running_k_hop == 1 and orig_src_idx_val != -1 and orig_src_idx_val != curr_src_idx_val:
                         if curr_src_idx_val in map_curr_src_to_local: # 確保 curr_src_idx_val 在映射中
                            path_elements_embs.append(current_path_src_embs_for_encoding[map_curr_src_to_local[curr_src_idx_val]])
                    
                    if path_elements_embs: # 確保序列不為空
                        src_sequences_for_transformer.append(torch.stack(path_elements_embs))
                    else: # 如果由於某種原因序列仍為空，添加一個占位符
                        src_sequences_for_transformer.append(torch.zeros((1, self.hdim), device=self.device))


                if not src_sequences_for_transformer:
                     src_b = torch.empty(0,0,self.hdim, device=self.device) 
                else:
                    src_b = pad_sequence(src_sequences_for_transformer, batch_first=True, padding_value=0.0)
            else: # MLP Encoder Logic
                path_specific_src_embs_for_mlp = current_path_src_embs_for_encoding[
                    torch.tensor([map_global_src_idx_to_local_in_unique[idx.item()] for idx in cand_src_idx_hop],
                                 dtype=torch.long, device=self.device)
                ]
                src_b = path_specific_src_embs_for_mlp[s_]

            # --- 後續的編碼與排序邏輯 ---
            # 準備 combined_tgt_edge_b (這部分需要小心，確保索引 s_ 的應用正確)
            map_global_tgt_idx_to_local_in_unique = {glob_idx.item(): i for i, glob_idx in enumerate(unique_hop_tgt_indices)}
            
            # cand_tgt_idx_hop[s_] 獲取當前 mini-batch 對應的目標節點的全局索引
            current_batch_tgt_global_indices = cand_tgt_idx_hop[s_]
            # 將這些全局索引轉換為在 unique_tgt_embs 中的局部索引
            current_batch_tgt_local_indices = torch.tensor(
                [map_global_tgt_idx_to_local_in_unique[idx.item()] for idx in current_batch_tgt_global_indices],
                dtype=torch.long, device=self.device
            )
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

    
class TestRel1PropagationRetrieve(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n--- Preparing Test Environment for Rel1 Propagation ---")
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        g = nx.DiGraph()
        # C01(0) --R1(0)--> C02(1)
        # C01(0) --R2(1)--> C03(2)
        # C02(1) --R3(2)--> C04(3)
        g.add_edge("C01", "C02", label="R1") 
        g.add_edge("C01", "C03", label="R2") 
        g.add_edge("C02", "C04", label="R3") 
        cls.tensor_graph = preprocess_graph_to_tensors(g)
        # Expected cui_to_idx: {'C01': 0, 'C02': 1, 'C03': 2, 'C04': 3}
        # Expected edge_to_idx (order might vary based on sorted labels): 
        # Assume R1:0, R2:1, R3:2 for this test
        print(f"Test graph edge_to_idx: {cls.tensor_graph['edge_to_idx']}")


    def assertTensorsEqual(self, t1, t2, msg=None):
        if t1 is None and t2 is None:
            return
        if t1 is None or t2 is None:
            self.fail(msg=f"{msg}\nOne tensor is None, the other is not.\nExpected: {t2}\nGot:      {t1}")
        self.assertTrue(torch.equal(t1.cpu(), t2.cpu()), msg=f"{msg}\nExpected: {t2.cpu()}\nGot:      {t1.cpu()}")

    def test_01_retrieve_first_hop_rel1_memory(self):
        print("\n--- Test 1: retrieve_neighbors - First Hop Rel1 Memory ---")
        src_cuis = ['C01']
        
        # For first hop, prev_candidate_tensors is None
        # Expected output: mem_first_edge_idx_hop should be tensor of -1s
        # as there's no "previous" first edge.
        s, t, e, mem_orig_s, mem_first_e = retrieve_neighbors_paths_no_self_tensorized(
            src_cuis, self.tensor_graph, None
        )
        
        c2i = self.tensor_graph['cui_to_idx']
        e2i = self.tensor_graph['edge_to_idx']

        # Paths: C01->C02, C01->C03
        expected_s = torch.tensor([c2i['C01'], c2i['C01']], device=self.device)
        # For the first hop, mem_first_e should indicate no prior first edge, e.g., filled with -1
        expected_mem_first_e = torch.tensor([-1, -1], dtype=torch.long, device=self.device)
        
        self.assertTensorsEqual(mem_first_e, expected_mem_first_e, "First hop's mem_first_edge should be -1s")
        print("PASS: retrieve_neighbors - First hop Rel1 memory is correctly -1.")

    def test_02_retrieve_second_hop_rel1_memory(self):
        print("\n--- Test 2: retrieve_neighbors - Second Hop Rel1 Memory (Crucial Test) ---")
        c2i = self.tensor_graph['cui_to_idx']
        e2i = self.tensor_graph['edge_to_idx'] # R1 should be e2i['R1']

        # Simulate state after first hop: C01 --R1--> C02 was selected
        # So, the "first_hop_edge" that led to C02 was R1.
        prev_state = {
            "selected_src_orig_idx": torch.tensor([c2i['C01']], device=self.device),
            "selected_hop_target_idx": torch.tensor([c2i['C02']], device=self.device),
            "selected_first_hop_edge_idx": torch.tensor([e2i['R1']], device=self.device) # Edge R1 led to C02
        }
        current_hop_start_cuis = ['C02'] # Second hop starts from C02

        s, t, e, mem_orig_s, mem_first_e = retrieve_neighbors_paths_no_self_tensorized(
            current_hop_start_cuis, self.tensor_graph, prev_state
        )

        # Expected path: C02 --R3--> C04
        # Original source was C01.
        # The first_hop_edge that led to the start of THIS hop (C02) was R1.
        expected_mem_orig_s = torch.tensor([c2i['C01']], device=self.device)
        expected_mem_first_e = torch.tensor([e2i['R1']], device=self.device) # This should carry over R1

        self.assertTensorsEqual(mem_orig_s, expected_mem_orig_s, "Second hop's mem_orig_src is incorrect")
        self.assertTensorsEqual(mem_first_e, expected_mem_first_e, "Second hop's mem_first_edge did not carry over Rel1 correctly")
        print("PASS: retrieve_neighbors - Second hop Rel1 memory propagated correctly.")

class TestGraphModelRel1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Similar setup as TestRetrieveNeighborsTensorized for graph and embeddings
        # Instantiate a mock CuiEmbedding, EdgeOneHot
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.hdim = 768 # Or your actual hdim
        g = nx.DiGraph()
        g.add_edge("C01", "C02", label="R1") 
        g.add_edge("C01", "C03", label="R2") 
        g.add_edge("C02", "C04", label="R3") 
        cls.g_nx_for_graphmodel = g
        mock_embedding_data = {cui: torch.randn(1, cls.hdim).to(cls.device) for cui in ["C01","C02","C03","C04"]}
        cls.mock_cui_lookup_for_graphmodel = MagicMock()
        cls.mock_cui_lookup_for_graphmodel.encode.side_effect = lambda cuis: torch.cat([mock_embedding_data[c] for c in cuis]).to(cls.device)
        cls.mock_cui_lookup_for_graphmodel.hdim = cls.hdim # Add hdim if your CuiEmbedding uses it
        cls.mock_cui_lookup_for_graphmodel.device = cls.device

        # Mock task and context embeddings
        cls.task_emb = torch.randn(1, cls.hdim, device=cls.device)
        cls.context_emb = torch.randn(1, cls.hdim, device=cls.device)
        
    def assertTensorsEqual(self, t1, t2, msg=None): # ## ADD THIS METHOD ##
        if t1 is None and t2 is None:
            return
        if t1 is None or t2 is None:
            self.fail(msg=f"{msg}\nOne tensor is None, the other is not.\nExpected: {t2}\nGot:      {t1}")
        self.assertTrue(torch.equal(t1.cpu(), t2.cpu()), msg=f"{msg}\nExpected: {t2.cpu()}\nGot:      {t1.cpu()}")


    def test_03_graphmodel_one_iteration_rel1_propagation(self):
        print("\n--- Test 3: GraphModel.one_iteration - Rel1 Propagation in next_hop_state ---")
        c2i = preprocess_graph_to_tensors(self.g_nx_for_graphmodel)['cui_to_idx'] # For consistency
        e2i = preprocess_graph_to_tensors(self.g_nx_for_graphmodel)['edge_to_idx']


        model = GraphModel(
            g_nx=self.g_nx_for_graphmodel,
            cui_embedding_lookup=self.mock_cui_lookup_for_graphmodel,
            hdim=self.hdim,
            nums_of_head=3, num_hops=2, top_n=1, device=self.device,
            path_encoder_type="MLP", # Keep it simple for this test
            # Ensure other GNN params are provided if your __init__ requires them
        )
        model.to(self.device)

        # --- First Hop ---
        _, _, first_hop_next_state, _ = model.one_iteration(
            task_emb_batch=self.task_emb,
            current_cui_str_list=['C01'],
            running_k_hop=0,
            context_emb_batch=self.context_emb,
            prev_iteration_state=None
        )
        self.assertIn("selected_first_hop_edge_idx", first_hop_next_state)
        self.assertIsNotNone(first_hop_next_state["selected_first_hop_edge_idx"])
        
        # Assuming C01->C02 (edge R1) is selected due to top_n=1 and some mock ranking
        # This part is tricky without actual ranking. We'll check if *an* edge is passed.
        # For a more robust test, you might need to mock p_ranker to force selection.
        selected_target_cui_idx_hop1 = first_hop_next_state["selected_hop_target_idx"][0].item()
        selected_first_edge_idx_hop1 = first_hop_next_state["selected_first_hop_edge_idx"][0].item()

        # Based on mock graph, if target is C02 (idx 1), edge should be R1 (idx from e2i['R1'])
        # If target is C03 (idx 2), edge should be R2 (idx from e2i['R2'])
        expected_edge_for_C02 = e2i['R1']
        self.assertTrue(
            (selected_target_cui_idx_hop1 == c2i['C02'] and selected_first_edge_idx_hop1 == expected_edge_for_C02) or \
            (selected_target_cui_idx_hop1 == c2i['C03'] and selected_first_edge_idx_hop1 == e2i['R2']),
            "First hop next_state did not capture the correct first edge."
        )
        print(f"First hop next_state contains selected_first_hop_edge_idx: {first_hop_next_state['selected_first_hop_edge_idx']}")


        # --- Second Hop ---
        # Assume the first hop selected C01 --R1--> C02
        prev_state_for_hop2 = {
            "selected_src_orig_idx": torch.tensor([c2i['C01']], device=self.device),
            "selected_hop_target_idx": torch.tensor([c2i['C02']], device=self.device),
            "selected_first_hop_edge_idx": torch.tensor([e2i['R1']], device=self.device)
        }

        all_paths_info_hop2, _, second_hop_next_state, _ = model.one_iteration(
            task_emb_batch=self.task_emb,
            current_cui_str_list=['C02'],
            running_k_hop=1,
            context_emb_batch=self.context_emb,
            prev_iteration_state=prev_state_for_hop2
        )
        
        # Check if selected_first_hop_edge_idx is propagated to the next_hop_state of the second hop
        self.assertIn("selected_first_hop_edge_idx", second_hop_next_state)
        self.assertTensorsEqual(
            second_hop_next_state["selected_first_hop_edge_idx"],
            torch.tensor([e2i['R1']], device=self.device), # Should still be R1
            "Second hop next_state did not correctly propagate selected_first_hop_edge_idx"
        )
        
        # Also check all_paths_info from the second hop
        self.assertIn("mem_first_edge_idx", all_paths_info_hop2)
        # Path found: C02->C04, its mem_first_edge_idx should be R1
        self.assertTensorsEqual(
            all_paths_info_hop2["mem_first_edge_idx"],
            torch.tensor([e2i['R1']], device=self.device),
            "Second hop all_paths_info did not correctly store mem_first_edge_idx"
        )
        print("PASS: GraphModel.one_iteration - Rel1 propagation in next_hop_state and all_paths_info correct.")
        
        
        
class TestGraphModelTransformerInput(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n--- Preparing Test Environment for Transformer Input Validation ---")
        cls.hdim = 768 # Example hidden dimension
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"TestGraphModelTransformerInput using device: {cls.device}")

        # 1. Create a simple NetworkX graph for testing
        cls.g_nx = nx.DiGraph()
        # C01(0) --R1(idx for R1)--> C02(1)
        # C01(0) --R2(idx for R2)--> C03(2)  (Alternative first hop)
        # C02(1) --R3(idx for R3)--> C04(3)
        cls.g_nx.add_edge("C01", "C02", label="R1")
        cls.g_nx.add_edge("C01", "C03", label="R2")
        cls.g_nx.add_edge("C02", "C04", label="R3")
        cls.nodes = sorted(list(cls.g_nx.nodes()))

        # 2. Preprocess graph (ensure your actual function is used)
        # This will define cui_to_idx and edge_to_idx used in tests
        cls.tensor_graph = preprocess_graph_to_tensors(cls.g_nx)
        cls.c2i = cls.tensor_graph['cui_to_idx']
        cls.e2i = cls.tensor_graph['edge_to_idx']
        print(f"Mock graph edge_to_idx for tests: {cls.e2i}")


        # 3. Mock CuiEmbedding object
        cls.mock_embedding_data = {
            cui: torch.randn(cls.hdim).to(cls.device) for cui in cls.nodes # Store as [hdim]
        }
        # Adjust mock_cui_lookup if your CuiEmbedding.encode returns different shape
        cls.mock_cui_lookup = MockCuiEmbedding(cls.mock_embedding_data, cls.hdim, cls.device)


        # 4. Task and Context Embeddings
        cls.task_emb = torch.randn(1, cls.hdim, device=cls.device)
        cls.context_emb = torch.randn(1, cls.hdim, device=cls.device)

    def assertTensorsEqual(self, t1, t2, msg=None):
        if t1 is None and t2 is None: return
        if t1 is None or t2 is None:
            self.fail(msg=f"{msg}\nOne tensor is None, other is not.\nExpected: {t2}\nGot: {t1}")
        self.assertTrue(torch.equal(t1.cpu(), t2.cpu()), msg=f"{msg}\nExpected: {t2.cpu()}\nGot: {t1.cpu()}")
    
    # We need to patch the *actual* PathEncoderTransformer that GraphModel instantiates
    # This requires knowing the module path if it's in a different file.
    # If it's in the same file (like here), '__main__.PathEncoderTransformer' is common.
    @patch('__main__.PathEncoderTransformer.forward')
    def test_04_transformer_encoder_input_sequence(self, mock_actual_transformer_forward):
        print("\n--- Test 4: PathEncoderTransformer Input Sequence (Rel1 included) ---")

        # Configure mock to return a correctly shaped tensor to allow p_ranker to proceed
        # The first dimension must match the batch size of paths fed to it.
        # We will determine the number of paths later. For now, set a side_effect.
        def mock_forward_side_effect(src_seq_batch, tgt_edge_combo_batch):
            # src_seq_batch shape: [num_paths_in_minibatch, max_seq_len, hdim]
            # tgt_edge_combo_batch shape: [num_paths_in_minibatch, hdim + edge_dim_actual]
            # PathEncoderTransformer should output [num_paths_in_minibatch, hdim]
            num_paths = src_seq_batch.shape[0]
            return torch.randn(num_paths, self.hdim, device=self.device)
        mock_actual_transformer_forward.side_effect = mock_forward_side_effect

        model = GraphModel(
            g_nx=self.g_nx,
            cui_embedding_lookup=self.mock_cui_lookup,
            hdim=self.hdim,
            nums_of_head=3, # Example
            num_hops=2,
            top_n=2, # Select up to 2 paths
            device=self.device,
            path_encoder_type="Transformer" # CRITICAL: Enable Transformer
            # Provide other necessary __init__ args for your GraphModel if any
        )
        # Replace the p_encoder's forward method with our mock *after* GraphModel init
        model.p_encoder.forward = mock_actual_transformer_forward
        model.to(self.device)

        # --- Simulate First Hop: C01 -> C02 (Rel1) and C01 -> C03 (Rel2) ---
        # We want to force selection of C01->C02 so that its Rel1 is R1.
        # This requires mocking p_ranker if top_n is small.
        # For simplicity, let top_n=2 to select both, or assume a specific order.
        
        # To make the selection deterministic for testing Rel1 propagation,
        # let's mock the p_ranker to score C01->C02 path higher if needed,
        # or simply ensure top_n >= number of 1st hop paths from C01.
        # Our mock graph from C01 has 2 paths. top_n=2 will select both.
        
        all_paths_info_h1, _, next_hop_state_h1, _ = model.one_iteration(
            task_emb_batch=self.task_emb,
            current_cui_str_list=['C01'],
            running_k_hop=0,
            context_emb_batch=self.context_emb,
            prev_iteration_state=None
        )
        self.assertIsNotNone(next_hop_state_h1, "Next_hop_state_h1 should not be None")
        self.assertIn("selected_first_hop_edge_idx", next_hop_state_h1, "selected_first_hop_edge_idx missing from hop1 state")
        self.assertIsNotNone(next_hop_state_h1["selected_first_hop_edge_idx"], "selected_first_hop_edge_idx is None in hop1 state")


        # --- Simulate Second Hop ---
        # Start from C02 (which was reached via R1 from C01)
        # The prev_iteration_state should be correctly formed by the first hop's next_hop_state.
        # Specifically, next_hop_state_h1["selected_first_hop_edge_idx"] should contain R1's index
        # for the path that targeted C02.

        # Let's assume C01->C02 was the first path and C01->C03 was the second.
        # And that top_n=2 selected both.
        # So, next_hop_state_h1['selected_hop_target_idx'] could be [idx(C02), idx(C03)]
        # And next_hop_state_h1['selected_first_hop_edge_idx'] could be [idx(R1), idx(R2)]

        # We only want to expand from C02 for this specific test point
        idx_of_C02_in_hop1_targets = -1
        for i, tgt_idx in enumerate(next_hop_state_h1['selected_hop_target_idx'].tolist()):
            if tgt_idx == self.c2i['C02']:
                idx_of_C02_in_hop1_targets = i
                break
        self.assertNotEqual(idx_of_C02_in_hop1_targets, -1, "C02 was not among selected targets from hop 1")

        prev_state_for_hop2_from_C02 = {
            "selected_src_orig_idx": next_hop_state_h1['selected_src_orig_idx'][idx_of_C02_in_hop1_targets].unsqueeze(0),
            "selected_hop_target_idx": next_hop_state_h1['selected_hop_target_idx'][idx_of_C02_in_hop1_targets].unsqueeze(0),
            "selected_first_hop_edge_idx": next_hop_state_h1['selected_first_hop_edge_idx'][idx_of_C02_in_hop1_targets].unsqueeze(0)
        }

        model.one_iteration(
            task_emb_batch=self.task_emb,
            current_cui_str_list=['C02'], # Expanding only from C02
            running_k_hop=1,
            context_emb_batch=self.context_emb,
            prev_iteration_state=prev_state_for_hop2_from_C02
        )

        self.assertTrue(mock_actual_transformer_forward.called, "PathEncoderTransformer's forward was not called in 2nd hop.")
        
        # Get arguments from the last call (assuming only one mini-batch for C02->C04 path)
        call_args, _ = mock_actual_transformer_forward.call_args
        src_input_to_transformer = call_args[0] # This is src_b

        # Path C01 --R1--> C02 --R3--> C04
        # When processing the second hop (C02 --R3--> C04):
        # Input sequence should be [emb(C01), projected_emb(R1), emb(C02)]
        # Sequence length = 3
        # Batch size for this mini-batch is 1 (only one path C02->C04)
        
        expected_shape_hop2 = (1, 3, self.hdim) # (batch_size=1 path, seq_len=3, hdim)
        self.assertEqual(src_input_to_transformer.shape, expected_shape_hop2,
                         f"Transformer input sequence shape for 2nd hop should be {expected_shape_hop2}, but got {src_input_to_transformer.shape}")
        print("PASS: PathEncoderTransformer input sequence shape for 2nd hop (with Rel1) is correct.")

        # Now, let's test a first hop call to the Transformer to ensure seq_len is 1
        mock_actual_transformer_forward.reset_mock() # Reset call history

        model.one_iteration(
            task_emb_batch=self.task_emb,
            current_cui_str_list=['C01'], # Expanding from C01
            running_k_hop=0, # First hop
            context_emb_batch=self.context_emb,
            prev_iteration_state=None
        )
        self.assertTrue(mock_actual_transformer_forward.called, "PathEncoderTransformer's forward was not called in 1st hop test.")
        call_args_hop1, _ = mock_actual_transformer_forward.call_args
        src_input_to_transformer_hop1 = call_args_hop1[0]

        # For the first hop (e.g., C01->C02), sequence is just [emb(C01)], so seq_len=1
        # There are two paths from C01, so batch_size=2 for this mini-batch (assuming path_per_batch_size >= 2)
        expected_shape_hop1 = (2, 1, self.hdim) # (batch_size=2 paths, seq_len=1, hdim)
        self.assertEqual(src_input_to_transformer_hop1.shape, expected_shape_hop1,
                         f"Transformer input sequence shape for 1st hop should be {expected_shape_hop1}, but got {src_input_to_transformer_hop1.shape}")
        print("PASS: PathEncoderTransformer input sequence shape for 1st hop is correct.")


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestRel1PropagationRetrieve))
    suite.addTest(unittest.makeSuite(TestGraphModelRel1)) # Uncomment when GraphModel and its deps are pasted
    suite.addTest(unittest.makeSuite(TestGraphModelTransformerInput))
    runner = unittest.TextTestRunner()
    runner.run(suite)
