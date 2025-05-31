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

# --- 重構後的 retrieve_neighbors_paths_no_self_tensorized ---
def retrieve_neighbors_paths_no_self_tensorized(
    current_cui_str_list, 
    tensor_graph, 
    prev_candidate_tensors=None
):
   
    if not current_cui_str_list:
        return torch.tensor([], dtype=torch.long), \
               torch.tensor([], dtype=torch.long), \
               torch.tensor([], dtype=torch.long), \
               None

    cui_to_idx = tensor_graph['cui_to_idx']
    
    # Convert current CUI strings to indices
    current_cui_indices = []
    for cui_str in current_cui_str_list:
        if cui_str in cui_to_idx:
            current_cui_indices.append(cui_to_idx[cui_str])
    
    if not current_cui_indices:
        return torch.tensor([], dtype=torch.long), \
               torch.tensor([], dtype=torch.long), \
               torch.tensor([], dtype=torch.long), \
               None
    device = tensor_graph['adj_src'].device         
    current_cui_indices_tensor = torch.tensor(list(set(current_cui_indices)), dtype=torch.long, device=device) # Unique current sources

    
    path_src_list = []
    path_tgt_list = []
    path_edge_type_list = []
    path_memory_src_prev_hop_list = [] # For 2+ hop tracking

    adj_src_graph = tensor_graph['adj_src']
    adj_tgt_graph = tensor_graph['adj_tgt']
    adj_edge_type_graph = tensor_graph['adj_edge_type']

    for i, src_idx_current_hop in enumerate(current_cui_indices_tensor):
        # Find all edges in the graph where the source is src_idx_current_hop

        mask_src_is_current = (adj_src_graph == src_idx_current_hop)
        if not torch.any(mask_src_is_current): # No outgoing edges from this source
            continue

        srcs_for_paths = adj_src_graph[mask_src_is_current] # These will all be src_idx_current_hop
        tgts_for_paths = adj_tgt_graph[mask_src_is_current]
        edge_types_for_paths = adj_edge_type_graph[mask_src_is_current]
        
        path_src_list.append(srcs_for_paths)
        path_tgt_list.append(tgts_for_paths)
        path_edge_type_list.append(edge_types_for_paths)



        if prev_candidate_tensors is not None and \
           'selected_src_orig_idx' in prev_candidate_tensors and \
           'selected_hop_target_idx' in prev_candidate_tensors:
            
            prev_tgt_mask = (prev_candidate_tensors['selected_hop_target_idx'] == src_idx_current_hop)
            if torch.any(prev_tgt_mask):
                # 確保 prev_candidate_tensors['selected_src_orig_idx'] 不為 None
                if prev_candidate_tensors['selected_src_orig_idx'] is not None:
                    original_source_for_this_src_candidates = prev_candidate_tensors['selected_src_orig_idx'][prev_tgt_mask]
                    if original_source_for_this_src_candidates.numel() > 0:
                        original_source_for_this_src = original_source_for_this_src_candidates[0] # Take the first one
                        path_memory_src_prev_hop_list.append(original_source_for_this_src.repeat(srcs_for_paths.size(0)))
                    else: # 理論上 mask_in_prev_targets 為 True 時，這裡不應為空
                         path_memory_src_prev_hop_list.append(src_idx_current_hop.repeat(srcs_for_paths.size(0))) # Fallback
                else: # selected_src_orig_idx is None (可能發生在第一跳傳給第二跳時，如果第一跳的記憶是 None)
                    path_memory_src_prev_hop_list.append(src_idx_current_hop.repeat(srcs_for_paths.size(0))) # Fallback
            else:
                # Fallback: use src_idx_current_hop if original source cannot be traced
                path_memory_src_prev_hop_list.append(src_idx_current_hop.repeat(srcs_for_paths.size(0)))
        else: # 第一跳，或者 prev_candidate_tensors 結構不對
            to_append = src_idx_current_hop.repeat(srcs_for_paths.size(0))
            path_memory_src_prev_hop_list.append(to_append)


    if not path_src_list: # No paths found at all
        return torch.tensor([], dtype=torch.long), \
               torch.tensor([], dtype=torch.long), \
               torch.tensor([], dtype=torch.long), \
               None

    candidate_src_indices = torch.cat(path_src_list)
    candidate_tgt_indices = torch.cat(path_tgt_list)
    candidate_edge_type_indices = torch.cat(path_edge_type_list)
    
 
    path_memory_src_prev_hop_tensor = torch.cat(path_memory_src_prev_hop_list) if path_memory_src_prev_hop_list else None
    
    # Filter out self-loops (src == tgt for the current hop)
    # This was part of "no_self" in the original name
    non_self_loop_mask = (candidate_src_indices != candidate_tgt_indices)
    candidate_src_indices = candidate_src_indices[non_self_loop_mask].to(device)
    candidate_tgt_indices = candidate_tgt_indices[non_self_loop_mask].to(device)
    candidate_edge_type_indices = candidate_edge_type_indices[non_self_loop_mask].to(device)
    if path_memory_src_prev_hop_tensor is not None:
        path_memory_src_prev_hop_tensor = path_memory_src_prev_hop_tensor[non_self_loop_mask].to(device)

    return candidate_src_indices, candidate_tgt_indices, candidate_edge_type_indices, path_memory_src_prev_hop_tensor




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
        
        

        if self.p_encoder_type == "Transformer":
            self.p_encoder = PathEncoderTransformer(hdim, hdim + actual_edge_dim)
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
        cand_src_idx_hop, cand_tgt_idx_hop, cand_edge_idx_hop, mem_orig_src_idx_hop = \
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
            
            # ## MODIFIED: 根據 p_encoder_type 準備不同的源嵌入
            if self.p_encoder_type == "Transformer":
                # --- 為 Transformer 準備序列輸入 ---
                src_sequences_for_transformer = []
                # 建立從全局索引到其在 unique 嵌入張量中位置的映射
                map_mem_orig_to_local = {glob_idx.item(): i for i, glob_idx in enumerate(unique_mem_orig_src_indices)} if unique_mem_orig_src_indices is not None else {}
                map_curr_src_to_local = {glob_idx.item(): i for i, glob_idx in enumerate(unique_hop_src_indices)}
                
                # 遍歷小批次中的每一條路徑
                for j in range(s_.start, s_.stop):
                    orig_src_idx = mem_orig_src_idx_hop[j].item()
                    curr_src_idx = cand_src_idx_hop[j].item()
                    
                    # 獲取嵌入
                    orig_src_emb = unique_mem_orig_src_embs[map_mem_orig_to_local[orig_src_idx]]
                    curr_src_emb = current_path_src_embs_for_encoding[map_curr_src_to_local[curr_src_idx]]

                    # 如果是第二跳或更深（初始節點和中間節點不同），則構建序列
                    if orig_src_idx != curr_src_idx:
                        src_sequences_for_transformer.append(torch.stack([orig_src_emb, curr_src_emb]))
                    else: # 如果是第一跳，歷史只有一個節點
                        src_sequences_for_transformer.append(curr_src_emb.unsqueeze(0))

                # 對序列進行填充，使其長度一致
                path_specific_src_embs_sequence = torch.nn.utils.rnn.pad_sequence(src_sequences_for_transformer, batch_first=True, padding_value=0.0)
                src_b = path_specific_src_embs_sequence
            else: # MLP 版本邏輯
                path_specific_src_embs = current_path_src_embs_for_encoding[
                    torch.tensor([map_global_src_idx_to_local_in_unique[idx.item()] for idx in cand_src_idx_hop], dtype=torch.long, device=self.device)
                ]
                src_b = path_specific_src_embs[s_]
                
             # --- 後續的編碼與排序邏輯 ---
            map_global_tgt_idx_to_local_in_unique = {glob_idx.item(): i for i, glob_idx in enumerate(unique_hop_tgt_indices)}
            path_specific_tgt_embs = unique_tgt_embs[torch.tensor([map_global_tgt_idx_to_local_in_unique[idx.item()] for idx in cand_tgt_idx_hop], dtype=torch.long, device=self.device)]
            combined_tgt_edge_embs_for_path_enc = torch.cat((path_specific_tgt_embs, path_edge_embs_for_gin_and_path_enc), dim=-1)
            combined_tgt_edge_b = combined_tgt_edge_embs_for_path_enc[s_]
            
            encoded_b = self.p_encoder(src_b, combined_tgt_edge_b)
            if self.p_encoder_type == "Transformer" and encoded_b.dim() == 3:
                encoded_b = encoded_b.squeeze(1)
            all_encoded_paths_hop.append(encoded_b)
            
            task_exp_b = task_emb_batch.expand(encoded_b.size(0), -1)
            ctx_exp_b = context_emb_batch.expand(encoded_b.size(0), -1) if context_emb_batch is not None else task_exp_b
            scores_b = self.p_ranker(task_exp_b, ctx_exp_b, encoded_b)
            all_path_scores_hop.append(scores_b)   

        if not all_path_scores_hop: return None, {}, None, None, True
        final_scores_hop = torch.cat(all_path_scores_hop, dim=0)
        encoded_paths_tensor_hop = torch.cat(all_encoded_paths_hop, dim=0)

        # --- 6. Top-N Selection ---
        top_n_val = min(self.top_n, final_scores_hop.size(0))
        if top_n_val == 0: return None, {}, None, None, True # No paths to select if final_scores_hop is empty
        
        _, top_k_indices = torch.topk(final_scores_hop.squeeze(-1), top_n_val, dim=0)
        
        sel_tgt_idx = cand_tgt_idx_hop[top_k_indices]
        sel_src_idx_thishop = cand_src_idx_hop[top_k_indices] # For debug/path string
        sel_edge_idx = cand_edge_idx_hop[top_k_indices] # For debug/path string
        sel_mem_orig_src_idx = mem_orig_src_idx_hop[top_k_indices] if mem_orig_src_idx_hop is not None else None

        all_paths_info = {"scores": final_scores_hop, "encoded_embeddings": encoded_paths_tensor_hop,
                          "src_idx": cand_src_idx_hop, "tgt_idx": cand_tgt_idx_hop,
                          "edge_idx": cand_edge_idx_hop, "mem_orig_src_idx": mem_orig_src_idx_hop}
        next_hop_state = {"selected_src_orig_idx": sel_mem_orig_src_idx,
                          "selected_hop_target_idx": sel_tgt_idx}
        
        # Path string construction (for debug, CPU-bound)
        visited_paths_str_dict = {}
        # idx_to_cui = self.g_tensorized['idx_to_cui'] # Already defined
        # idx_to_edge = self.g_tensorized['idx_to_edge']
        # for i in range(sel_tgt_idx.size(0)):
        #     tgt_s = idx_to_cui.get(sel_tgt_idx[i].item())
        #     src_s = idx_to_cui.get(sel_src_idx_thishop[i].item()) # Src of current hop path
        #     edge_s = idx_to_edge.get(sel_edge_idx[i].item())
        #     if tgt_s: visited_paths_str_dict[tgt_s] = f"... -> {src_s} --({edge_s})--> {tgt_s}"
        
        return all_paths_info, visited_paths_str_dict, next_hop_state, stop_flag
    
    
class TestGraphModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """在所有測試開始前，準備一次共享資源"""
        print("\n--- 準備測試環境: 創建 Mock 圖譜和嵌入 ---")
        cls.hdim = 768
        cls.device = torch.device('cuda') # 在 CPU 上測試以簡化

        # 1. 創建一個簡單的 NetworkX 圖
        cls.g_nx = nx.DiGraph()
        cls.g_nx.add_edge("C01", "C02", label="R1")
        cls.g_nx.add_edge("C01", "C03", label="R2")
        cls.g_nx.add_edge("C02", "C04", label="R3") # 這構成了一條 2-hop 路徑: C01 -> C02 -> C04
        cls.nodes = sorted(list(cls.g_nx.nodes())) # ['C01', 'C02', 'C03', 'C04']

        # 2. 創建一個 Mock CuiEmbedding 對象
        mock_embedding_data = {cui: torch.randn(1, cls.hdim) for cui in cls.nodes}
        cls.mock_cui_lookup = MagicMock()
        cls.mock_cui_lookup.encode.side_effect = lambda cuis: torch.cat([mock_embedding_data[c] for c in cuis]).to(cls.device)
        
        # 3. 準備其他輸入
        cls.mock_cui_weights = {cui: 1.0 for cui in cls.nodes}
        cls.task_emb = torch.randn(1, cls.hdim, device=cls.device)
        cls.context_emb = torch.randn(1, cls.hdim, device=cls.device)

    def test_01_graphmodel_initialization(self):
        """測試 GraphModel 是否能根據配置正確初始化"""
        print("\n--- 測試 1: GraphModel 初始化 ---")
        model = GraphModel(
            g_nx=self.g_nx,
            cui_embedding_lookup=self.mock_cui_lookup,
            hdim=self.hdim,
            nums_of_head=4, num_hops=2, top_n=8, device=self.device,
            gnn_update=True, gnn_type="Stack" # 測試 GINStack
        )
        self.assertIsNotNone(model.g_tensorized, "圖譜張量化不應為 None")
        self.assertEqual(model.g_tensorized['num_nodes'], len(self.nodes), "張量化後的節點數量不對")
        self.assertIsInstance(model.gnn, GINStack, "GNN 類型應為 GINStack")
        print("PASS: GraphModel 初始化成功。")

    def test_02_one_iteration_first_hop(self):
        """測試第一跳推理的完整流程"""
        print("\n--- 測試 2: one_iteration 第一跳推理 ---")
        model = GraphModel(
            g_nx=self.g_nx, cui_embedding_lookup=self.mock_cui_lookup, hdim=self.hdim,
            nums_of_head=4, num_hops=2, top_n=8, device=self.device
        )
        model.to(self.device) # 確保模型在正確的設備上

        all_paths_info, _, next_hop_state, stop_flag = model.one_iteration(
            task_emb_batch=self.task_emb,
            current_cui_str_list=['C01'], # 從 C01 開始
            running_k_hop=0,
            context_emb_batch=self.context_emb,
            prev_iteration_state=None
        )

        self.assertFalse(stop_flag, "第一跳不應停止")
        self.assertIsNotNone(all_paths_info, "路徑資訊不應為 None")
        
        # 預期從 C01 出發有兩條路徑：到 C02 和 C03
        num_paths_found = all_paths_info['scores'].shape[0]
        self.assertEqual(num_paths_found, 2, "第一跳應找到2條路徑")

        # 檢查返回的張量形狀
        self.assertEqual(all_paths_info['encoded_embeddings'].shape, (2, self.hdim))
        self.assertEqual(all_paths_info['scores'].shape, (2, 1))

        # 檢查 Top-N 選擇後的結果
        # 因為 top_n=8 > 2，所以兩條路徑都應被選中
        selected_targets_idx = next_hop_state['selected_hop_target_idx']
        self.assertEqual(selected_targets_idx.shape[0], 2)
        
        # 將索引轉回 CUI 字串進行驗證
        idx_to_cui = model.g_tensorized['idx_to_cui']
        selected_targets_cui = {idx_to_cui[idx.item()] for idx in selected_targets_idx}
        self.assertSetEqual(selected_targets_cui, {'C02', 'C03'}, "被選中的目標節點應為 C02 和 C03")
        print("PASS: 第一跳推理流程正確。")

    def test_03_one_iteration_second_hop(self):
        """測試第二跳推理，特別是路徑歷史的傳遞"""
        print("\n--- 測試 3: one_iteration 第二跳推理 ---")
        model = GraphModel(
            g_nx=self.g_nx, cui_embedding_lookup=self.mock_cui_lookup, hdim=self.hdim,
            nums_of_head=4, num_hops=2, top_n=8, device=self.device
        )
        model.to(self.device)

        # 手動模擬第一跳的輸出，作為第二跳的輸入
        cui_to_idx = model.g_tensorized['cui_to_idx']
        first_hop_output = {
            "selected_src_orig_idx": torch.tensor([cui_to_idx['C01'], cui_to_idx['C01']], device=self.device),
            "selected_hop_target_idx": torch.tensor([cui_to_idx['C02'], cui_to_idx['C03']], device=self.device)
        }
        
        all_paths_info, _, _, stop_flag = model.one_iteration(
            task_emb_batch=self.task_emb,
            current_cui_str_list=['C02', 'C03'], # 從 C02 和 C03 開始第二跳
            running_k_hop=1,
            context_emb_batch=self.context_emb,
            prev_iteration_state=first_hop_output
        )

        self.assertFalse(stop_flag)
        
        # 預期：只有 C02 有出邊到 C04，所以只會找到一條路徑
        num_paths_found = all_paths_info['scores'].shape[0]
        self.assertEqual(num_paths_found, 1, "第二跳應只找到1條路徑 (C02->C04)")
        
        # **關鍵驗證**: 檢查路徑記憶是否正確
        # `mem_orig_src_idx` 應該是這條路徑的最初始源頭，即 C01
        mem_idx = all_paths_info['mem_orig_src_idx'][0].item()
        self.assertEqual(mem_idx, cui_to_idx['C01'], "路徑記憶應追蹤到最初始的源頭 C01")
        
        # 檢查這條路徑的源和目標
        src_idx = all_paths_info['src_idx'][0].item()
        tgt_idx = all_paths_info['tgt_idx'][0].item()
        self.assertEqual(src_idx, cui_to_idx['C02'], "第二跳路徑的源應為 C02")
        self.assertEqual(tgt_idx, cui_to_idx['C04'], "第二跳路徑的目標應為 C04")
        print("PASS: 第二跳推理及路徑歷史追蹤正確。")

    @patch('__main__.PathEncoderTransformer.forward') # 根據您實際的文件結構修改 '__main__'
    def test_04_transformer_encoder_input_shape(self, mock_transformer_forward):
        """測試當使用 Transformer 編碼器時，輸入序列的形狀是否正確"""
        print("\n--- 測試 4: PathEncoderTransformer 輸入序列驗證 ---")
        
        # 設置返回一個符合尺寸的假輸出以繼續流程
        mock_transformer_forward.return_value = torch.randn(1, self.hdim, device=self.device)

        model = GraphModel(
            g_nx=self.g_nx, cui_embedding_lookup=self.mock_cui_lookup, hdim=self.hdim,
            nums_of_head=4, num_hops=2, top_n=8, device=self.device,
            path_encoder_type="Transformer" # **啟用 Transformer**
        )
        model.p_encoder.forward = mock_transformer_forward # 替換為 mock
        model.to(self.device)

        # 模擬第二跳
        cui_to_idx = model.g_tensorized['cui_to_idx']
        first_hop_output = {
            "selected_src_orig_idx": torch.tensor([cui_to_idx['C01']], device=self.device),
            "selected_hop_target_idx": torch.tensor([cui_to_idx['C02']], device=self.device)
        }

        model.one_iteration(
            task_emb_batch=self.task_emb,
            current_cui_str_list=['C02'], # 只從 C02 擴展
            running_k_hop=1,
            context_emb_batch=self.context_emb,
            prev_iteration_state=first_hop_output
        )

        # 驗證 mock_transformer_forward 被調用
        self.assertTrue(mock_transformer_forward.called, "PathEncoderTransformer 的 forward 未被調用")
        
        # 獲取傳遞給 forward 的參數
        call_args, _ = mock_transformer_forward.call_args
        src_input_to_transformer = call_args[0] # forward 的第一個參數是 src

        # **關鍵驗證**: 檢查輸入序列的形狀
        # 因為是第二跳，歷史有2個節點（初始+中間），所以 seq_len 應為 2
        # 小批次大小為 1 (因為只找到一條 C02->C04 的路徑)
        expected_shape = (1, 2, self.hdim) # (batch_size, seq_len, hdim)
        self.assertEqual(src_input_to_transformer.shape, expected_shape, 
                         f"Transformer 的源輸入序列形狀應為 {expected_shape}，但得到 {src_input_to_transformer.shape}")
        print("PASS: PathEncoderTransformer 的輸入序列形狀正確。")


if __name__ == '__main__':
    # 為了在 notebook 或非標準環境下運行
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestGraphModel))
    runner = unittest.TextTestRunner()
    runner.run(suite)