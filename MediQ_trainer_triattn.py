# Required imports (ensure these are at the top of your file)

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
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=self.device)
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
        self.path_per_batch_size = 4096
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
            
        pruning_threshold_count = 4096 # 您設定的篩選路徑數量上限
        num_paths_this_hop_before_pruning = num_paths_this_hop # ## NOW THIS IS VALID ##

        if num_paths_this_hop > pruning_threshold_count: # Check against initial num_paths_this_hop
            # print(f"  Hop {running_k_hop}: Path count {num_paths_this_hop} exceeds threshold {pruning_threshold_count}. Applying pruning.")

            map_global_tgt_idx_to_local_in_unique = { # Ensure this map is created using pre-pruning unique_hop_tgt_indices
                glob_idx.item(): i for i, glob_idx in enumerate(unique_hop_tgt_indices)
            }
            # Ensure path_specific_tgt_local_indices can be created even if unique_tgt_embs was empty
            if unique_tgt_embs.numel() > 0 :
                path_specific_tgt_local_indices = torch.tensor(
                    [map_global_tgt_idx_to_local_in_unique[glob_idx.item()] for glob_idx in cand_tgt_idx_hop],
                    dtype=torch.long, device=self.device
                )
                path_specific_tgt_embs_for_pruning = unique_tgt_embs[path_specific_tgt_local_indices]

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
        
        # 從 current_path_src_embs_for_encoding (unique src embs 更新/未更新版) 中獲取每條剪枝後路徑的源嵌入
        path_specific_src_embs = current_path_src_embs_for_encoding[
            torch.tensor([map_global_src_idx_to_local_in_unique[idx.item()] for idx in cand_src_idx_hop],
                         dtype=torch.long, device=self.device)
        ]
        
        # 從 unique_tgt_embs 中獲取每條剪枝後路徑的目標嵌入
        # (map_global_tgt_idx_to_local_in_unique 需要在剪枝前就根據 unique_hop_tgt_indices 創建好)
        # 這裡的 unique_hop_tgt_indices 是剪枝前的，所以 map 也應該是剪枝前的
        # 如果剪枝了，cand_tgt_idx_hop 變短了，map_global_tgt_idx_to_local_in_unique 仍然是舊的
        # 我們需要的是剪枝後的 unique_hop_tgt_indices 對應的 map
        
        # 重新構建 path_specific_tgt_embs 和 combined_tgt_edge_embs_for_path_enc
        # 基於剪枝後的 cand_tgt_idx_hop 和 path_edge_embs_for_gin_and_path_enc
        
        # 獲取剪枝後路徑的目標節點的局部索引 (相對於 unique_tgt_embs)
        # unique_tgt_embs 是在剪枝前計算的，所以 map_global_tgt_idx_to_local_in_unique 仍然有效
        _map_global_tgt_idx_to_local_in_unique_for_encoding = {
            glob_idx.item(): i for i, glob_idx in enumerate(unique_hop_tgt_indices) # 使用剪枝前的 unique_hop_tgt_indices
        }
        path_specific_tgt_embs_for_encoding = unique_tgt_embs[
            torch.tensor([_map_global_tgt_idx_to_local_in_unique_for_encoding[idx.item()] for idx in cand_tgt_idx_hop], # 使用剪枝後的 cand_tgt_idx_hop
                         dtype=torch.long, device=self.device)
        ]
        
        # path_edge_embs_for_gin_and_path_enc 已經在步驟 4d 被剪枝了
        combined_tgt_edge_embs_for_path_enc = torch.cat(
            (path_specific_tgt_embs_for_encoding, path_edge_embs_for_gin_and_path_enc), dim=-1
        )

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
                src_sequences_for_transformer = []
                # 映射：全局索引 -> 在 unique 張量中的局部索引 (這些 map 基於剪枝前的 unique 索引)
                _map_mem_orig_to_local = {glob_idx.item(): local_idx for local_idx, glob_idx in enumerate(unique_mem_orig_src_indices)} if unique_mem_orig_src_indices is not None else {}
                _map_curr_src_to_local_for_transformer = map_global_src_idx_to_local_in_unique # 基於剪枝前的 unique_hop_src_indices

                for j_path_idx_in_batch in range(s_.start, s_.stop): # j_path_idx_in_batch 是剪枝後路徑列表的索引
                    path_elements_embs = []
                    
                    # 1. 添加最初始源節點嵌入 (CUI_A)
                    # mem_orig_src_idx_hop 此時是剪枝後的張量
                    orig_src_idx_val = mem_orig_src_idx_hop[j_path_idx_in_batch].item() if mem_orig_src_idx_hop is not None else -1
                    if orig_src_idx_val != -1 and orig_src_idx_val in _map_mem_orig_to_local and unique_mem_orig_src_embs is not None:
                        path_elements_embs.append(unique_mem_orig_src_embs[_map_mem_orig_to_local[orig_src_idx_val]])
                    else: 
                        # cand_src_idx_hop 此時是剪枝後的張量
                        curr_src_idx_val_for_fallback = cand_src_idx_hop[j_path_idx_in_batch].item()
                        if curr_src_idx_val_for_fallback in _map_curr_src_to_local_for_transformer :
                             path_elements_embs.append(current_path_src_embs_for_encoding[_map_curr_src_to_local_for_transformer[curr_src_idx_val_for_fallback]])

                    # 2. 添加第一個關係的嵌入 (Rel1) - 僅當是第二跳且 Rel1 存在時
                    if running_k_hop == 1 and mem_first_edge_idx_hop is not None:
                        # mem_first_edge_idx_hop 此時是剪枝後的張量
                        first_edge_idx_val = mem_first_edge_idx_hop[j_path_idx_in_batch].item()
                        if first_edge_idx_val != -1: 
                            if first_edge_idx_val < self.e_encoder.onehot_mat.shape[0]:
                                first_rel_one_hot = self.e_encoder.onehot_mat[first_edge_idx_val].to(self.device)
                                projected_first_rel_emb = self.edge_to_node_projection_for_transformer(first_rel_one_hot)
                                path_elements_embs.append(projected_first_rel_emb)
                            else: # 索引越界
                                path_elements_embs.append(torch.zeros(self.hdim, device=self.device))
                    
                    # 3. 添加中間節點嵌入 (CUI_B) - 僅當是第二跳且與最初始源節點不同時
                    # cand_src_idx_hop 此時是剪枝後的張量
                    curr_src_idx_val = cand_src_idx_hop[j_path_idx_in_batch].item()
                    if running_k_hop == 1 and orig_src_idx_val != -1 and orig_src_idx_val != curr_src_idx_val:
                         if curr_src_idx_val in _map_curr_src_to_local_for_transformer:
                            path_elements_embs.append(current_path_src_embs_for_encoding[_map_curr_src_to_local_for_transformer[curr_src_idx_val]])
                    
                    if path_elements_embs:
                        src_sequences_for_transformer.append(torch.stack(path_elements_embs))
                    else:
                        src_sequences_for_transformer.append(torch.zeros((1, self.hdim), device=self.device))

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


class Trainer(nn.Module):
    def __init__(self, tokenizer,
                 encoder, 
                 g_nx, 
                 cui_embedding_lookup, 
                 hdim,
                 nums_of_head, 
                 cui_vocab_str_to_idx, 
                 top_n, # top_n 仍然用於 GraphModel 內部決定下一跳的探索寬度
                 device,
                 nums_of_epochs,
                 LR,
                 cui_weights_dict=None,
                 contrastive_learning=True,
                 save_model_path=None,
                 gnn_update=True,
                 intermediate=False, 
                 score_threshold=0.5, # ## ADDED: 評分閾值
                 distance_metric="Cosine",
                 path_encoder_type="MLP",
                 path_ranker_type="Flat",
                 gnn_type="Stack",
                 gin_hidden_dim=None,
                 gin_num_layers=1,
                 triplet_margin=1.0,
                 early_stopping_patience=3,
                 early_stopping_metric='val_loss',
                 early_stopping_delta=0.001
                 ):
        super(Trainer, self).__init__()

        self.tokenizer = tokenizer
        self.encoder = encoder
        self.k_hops = 2 
        self.device = device
        self.cui_vocab_str_to_idx = cui_vocab_str_to_idx 
        self.rev_cui_vocab_idx_to_str = {v: k for k, v in cui_vocab_str_to_idx.items()}
        self.top_n_for_exploration = top_n # ## RENAMED for clarity

        self.gmodel = GraphModel(
            g_nx=g_nx, 
            cui_embedding_lookup=cui_embedding_lookup,
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
        self.weight_decay = 1e-4
        self.nums_of_epochs = nums_of_epochs
        self.intermediate = intermediate
        self.print_step = 50
        self.distance_metric = distance_metric
        self.mode = 'train'
        self.contrastive_learning = contrastive_learning
        self.triplet_margin = triplet_margin
        self.score_threshold = score_threshold # ## ADDED

        self.loss_fn_bce = nn.BCEWithLogitsLoss()
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


        print("**** ============= TRAINER (MediQ Tensorized GModel with Thresholding) ============= **** ")
        exp_setting = (f"TRAINER SETUP: SCORE_THRESHOLD: {self.score_threshold}\n"
                       f"INTERMEDIATE LOSS SUPERVISION: {self.intermediate}\n"
                       f"CONTRASTIVE LEARNING: {self.contrastive_learning}\n"
                       # ... (可以加入更多超參數到日誌中)
                      )
        logging.info(exp_setting)
        print(exp_setting)
        self.optimizer = None

    def create_optimizers(self):
                
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.encoder.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.encoder.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0},
            {'params': [p for n, p in self.gmodel.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.gmodel.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
        ]
        effective_lr = self.LR[0] if isinstance(self.LR, list) else self.LR
        print(f"Using Learning Rate: {effective_lr}")
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=effective_lr, eps=self.adam_epsilon)
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
        hop_loss = torch.tensor(0.0, device=self.device)
        if all_paths_info_hop is None or all_paths_info_hop['scores'] is None or \
           all_paths_info_hop['scores'].numel() == 0 or not gt_cuis_str_list_sample:
            return hop_loss

        path_scores = all_paths_info_hop['scores'].squeeze(-1) # [num_paths_this_hop]
        path_target_global_indices = all_paths_info_hop['tgt_idx'] # [num_paths_this_hop] (GLOBAL CUI INDICES)
        
        gt_global_indices_set = set(self._get_gt_indices_tensor(gt_cuis_str_list_sample).tolist())
        if not gt_global_indices_set: # No valid GT CUIs for this sample/hop
            return hop_loss

        # Aggregate scores for each unique target CUI index found in paths
        unique_path_targets_global_indices, inverse_indices = torch.unique(path_target_global_indices, return_inverse=True)
        
        # Use scatter_max to get the max score for each unique target CUI index
        # If torch_scatter not available, use a loop or other PyTorch methods
        if TORCH_SCATTER_AVAILABLE:
            # aggregated_scores_for_unique_targets = scatter_max(path_scores, inverse_indices, dim=0, dim_size=unique_path_targets_global_indices.size(0))[0]
             # scatter_max returns (values, argmax_indices). We only need values.
             # Fallback if scatter_max is not directly available or has issues with PyTorch versions
            temp_aggregated_scores = torch.full((unique_path_targets_global_indices.size(0),), float('-inf'), device=self.device, dtype=path_scores.dtype)
            for i, unique_tgt_idx in enumerate(unique_path_targets_global_indices):
                mask = (path_target_global_indices == unique_tgt_idx)
                if torch.any(mask):
                    temp_aggregated_scores[i] = torch.max(path_scores[mask])
            aggregated_scores_for_unique_targets = temp_aggregated_scores

        else: # Fallback without torch_scatter
            # print("BCE Fallback: Looped aggregation for scores.")
            aggregated_scores_for_unique_targets_list = []
            for unique_tgt_idx in unique_path_targets_global_indices.tolist():
                mask = (path_target_global_indices == unique_tgt_idx)
                if torch.any(mask):
                    aggregated_scores_for_unique_targets_list.append(torch.max(path_scores[mask]))
                # else: handle case where unique_tgt_idx somehow has no scores (should not happen)
            if not aggregated_scores_for_unique_targets_list: return hop_loss
            aggregated_scores_for_unique_targets = torch.stack(aggregated_scores_for_unique_targets_list)

        labels_for_unique_targets = torch.tensor(
            [1.0 if idx.item() in gt_global_indices_set else 0.0 for idx in unique_path_targets_global_indices],
            device=self.device
        )
        
        if aggregated_scores_for_unique_targets.numel() > 0 :
             hop_loss = self.loss_fn_bce(aggregated_scores_for_unique_targets, labels_for_unique_targets)
        return hop_loss


    def compute_triplet_loss_for_hop(self, anchor_embedding, all_paths_info_hop, gt_cuis_str_list_sample):
        triplet_loss = torch.tensor(0.0, device=self.device)
        if anchor_embedding is None or all_paths_info_hop is None or \
           all_paths_info_hop['encoded_embeddings'] is None or \
           all_paths_info_hop['encoded_embeddings'].numel() == 0 or not gt_cuis_str_list_sample:
            return triplet_loss

        path_embeddings = all_paths_info_hop['encoded_embeddings'] # [num_paths, hdim]
        path_target_global_indices = all_paths_info_hop['tgt_idx']   # [num_paths] (GLOBAL CUI INDICES)

        gt_global_indices_set = set(self._get_gt_indices_tensor(gt_cuis_str_list_sample).tolist())
        if not gt_global_indices_set: return triplet_loss
        
        positive_indices_mask = torch.tensor(
            [idx.item() in gt_global_indices_set for idx in path_target_global_indices],
            dtype=torch.bool, device=self.device
        )
        negative_indices_mask = ~positive_indices_mask

        positive_embs = path_embeddings[positive_indices_mask]
        negative_embs = path_embeddings[negative_indices_mask]

        if positive_embs.numel() == 0 or negative_embs.numel() == 0:
            return triplet_loss # Need at least one positive and one negative

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
        
        # ## MODIFIED: 用於存儲最終預測結果 (包含完整路徑信息)
        # 結構: batch_final_predictions[sample_idx] = { "path_tuple_str": "CUI_A->Rel1->CUI_B->Rel2->CUI_C", "final_target_cui": "CUI_C", "score": score, "hop": 2 }
        # 或者更簡單: batch_final_predictions[sample_idx] = [{"target": CUI_str, "hop": N, "score": S, "full_path_indices": (orig_s, r1, inter_s, r2, final_t)}]
        batch_final_predictions_for_acc = [[] for _ in range(batch_size)] 
        # 用於評估1跳的準確性
        batch_hop1_predictions_for_acc = [[] for _ in range(batch_size)] 

        for i in range(batch_size): # Sample 迴圈
            sample_loss_this_item = torch.tensor(0.0, device=self.device)
            task_emb_sample = input_task_embs_batch[i].unsqueeze(0) 
            known_cuis_str_sample = known_cuis_str_batch[i]
            context_emb_sample = self.compute_context_embedding(known_cuis_str_sample)
            if context_emb_sample is None: context_emb_sample = task_emb_sample

            current_cui_str_list_for_hop = known_cuis_str_sample
            prev_iter_state_for_next_hop = None
            
            # ## ADDED: 儲存當前樣本每一跳超過閾值的路徑信息
            # 格式: { path_tuple_key: {"target_idx": CUI_idx, "score": float, "hop": int, "path_indices": tuple} }
            # path_tuple_key 可以是 (orig_src_idx, first_edge_idx, intermediate_idx, second_edge_idx, final_target_idx)
            # 或簡化為 (orig_src_idx, intermediate_idx_or_final_target_idx) for 1-hop
            # (orig_src_idx, intermediate_idx, final_target_idx) for 2-hop
            # 為了路徑取代，我們需要能唯一識別路徑並追蹤其組成
            # 結構: high_confidence_paths_sample[hop_num] = list of dicts
            # dict = {"orig_src_idx": tensor, "first_edge_idx": tensor (or -1), "inter_or_final_tgt_idx": tensor, "final_tgt_idx": tensor (for 2hop), "score": tensor, "hop_num": int}
            
            # 簡化：只儲存用於下一跳探索的 top_n 節點的詳細信息
            # 和一個用於最終預測的、基於閾值的候選列表
            
            # 儲存每一跳超過閾值的 (最初始源頭CUI, 最終目標CUI, 跳數, 完整路徑元組(索引), 分數)
            # full_path_indices: (orig_s_idx, r1_idx, inter_s_idx, r2_idx, final_t_idx)
            # 對於1-hop: (orig_s_idx, r1_idx, final_t_idx, -1, -1)
            # 這裡的 key 可以是最終目標CUI，value是包含路徑和分數的字典，以處理多個路徑指向同一目標
            # 但為了路徑取代，我們需要以 "最初始源頭->中間節點" 作為鍵
            
            # 最終預測集合，key: (orig_src_idx, first_hop_target_idx), value: {"path": path_tuple, "score": score, "hop": 1 or 2}
            current_sample_final_preds_dict = {}


            for running_k in range(self.k_hops): # Hop 迴圈
                if not current_cui_str_list_for_hop and running_k > 0 : break
                
                all_paths_info_hop, _, \
                next_hop_state_info_for_exploration, stop_flag = self.gmodel.one_iteration(
                    task_emb_sample, current_cui_str_list_for_hop, running_k,
                    context_emb_sample, prev_iter_state_for_next_hop
                )

                if stop_flag or all_paths_info_hop is None: break

                # --- 損失計算 (GT選取邏輯已更新) ---
                gt_cuis_str_list_this_hop = []
                if running_k == 0:
                    gt_cuis_str_list_this_hop.extend(hop1_target_cuis_str_batch[i])
                    if self.intermediate:
                        gt_cuis_str_list_this_hop.extend(intermediate_target_cuis_batch[i])
                elif running_k == 1:
                    gt_cuis_str_list_this_hop.extend(hop2_target_cuis_str_batch[i])
                gt_cuis_str_list_this_hop = list(set(gt_cuis_str_list_this_hop))

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
                current_hop_total_loss = current_hop_bce_loss + current_hop_triplet_loss
                sample_loss_this_item = sample_loss_this_item + current_hop_total_loss
                
                # --- ## MODIFIED: 基於閾值篩選高質量預測，並執行路徑取代 ---
                path_scores = all_paths_info_hop['scores'].squeeze(-1) # [num_paths]
                confident_mask = (path_scores >= self.score_threshold)
                
                confident_path_indices = torch.where(confident_mask)[0]

                if confident_path_indices.numel() > 0:
                    conf_orig_srcs = all_paths_info_hop['mem_orig_src_idx'][confident_path_indices]
                    conf_first_edges = all_paths_info_hop['mem_first_edge_idx'][confident_path_indices] if all_paths_info_hop['mem_first_edge_idx'] is not None else None
                    conf_hop_srcs = all_paths_info_hop['src_idx'][confident_path_indices] # 當前跳的源 (中間節點 for hop2)
                    conf_hop_edges = all_paths_info_hop['edge_idx'][confident_path_indices] # 當前跳的邊 (Rel2 for hop2)
                    conf_hop_tgts = all_paths_info_hop['tgt_idx'][confident_path_indices]   # 當前跳的目標 (最終目標 for hop2)
                    conf_scores = path_scores[confident_path_indices]

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


                            # 記錄用於1-hop accuracy計算的目標
                            batch_hop1_predictions_for_acc[i].append(self.rev_cui_vocab_idx_to_str.get(final_t_idx))
                            
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
            
            accumulated_batch_loss = accumulated_batch_loss + sample_loss_this_item
            # 將 current_sample_final_preds_dict 中的 final_target_idx 收集起來用於 acc 計算
            batch_final_predictions_for_acc[i] = [
                self.rev_cui_vocab_idx_to_str.get(pred_info['final_target_idx'])
                for pred_info in current_sample_final_preds_dict.values()
                if self.rev_cui_vocab_idx_to_str.get(pred_info['final_target_idx']) is not None
            ]
            # 確保 batch_hop1_predictions_for_acc[i] 也是去重的字符串列表
            batch_hop1_predictions_for_acc[i] = list(set(batch_hop1_predictions_for_acc[i]))


        avg_batch_loss = accumulated_batch_loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=self.device)
        
        # final_predicted_cuis_str_batch 用於與 hop2_target_cuis 比較 (或者一個合併的GT)
        # batch_hop1_predictions_for_acc 用於與 hop1_target_cuis 比較
        # 這個返回值需要調整，以包含不同hop的預測
        return avg_batch_loss, batch_final_predictions_for_acc, batch_hop1_predictions_for_acc


    def measure_accuracy(self, final_predicted_cuis_str_batch, target_cuis_str_batch, mode="Recall@N"):
        # 這個函數需要被擴展或複製以處理不同 hop 的 accuracy
        # 例如，可以有一個 measure_recall, measure_precision, measure_f1
        # 這裡暫時保持原樣，但您可能需要為1-hop和最終結果分別調用它，或傳入mode
        batch_size = len(target_cuis_str_batch)
        if batch_size == 0: return 0.0, 0.0, 0.0 # P, R, F1
        
        all_precisions, all_recalls, all_f1s = [], [], []

        for i in range(batch_size):
            gold_cuis = set(target_cuis_str_batch[i])
            # final_predicted_cuis_str_batch[i] 應該是去重後的字符串列表
            pred_cuis = set(final_predicted_cuis_str_batch[i]) 
            
            num_pred = len(pred_cuis)
            num_gold = len(gold_cuis)
            num_intersect = len(gold_cuis.intersection(pred_cuis))

            precision = num_intersect / num_pred if num_pred > 0 else 0.0
            recall = num_intersect / num_gold if num_gold > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)
            
        avg_precision = np.mean(all_precisions) if all_precisions else 0.0
        avg_recall = np.mean(all_recalls) if all_recalls else 0.0
        avg_f1 = np.mean(all_f1s) if all_f1s else 0.0
        
        return avg_precision, avg_recall, avg_f1

    def train(self, train_data_loader, dev_data_loader, lr_scheduler=None):
        if self.optimizer is None: self.create_optimizers()
        update_step = 4 

        for ep in range(self.nums_of_epochs):
            print(f"\n--- Starting Epoch {ep+1}/{self.nums_of_epochs} ---")
            self.mode = 'train'; self.encoder.train(); self.gmodel.train()
            epoch_loss_train_list = []
            epoch_p1_train, epoch_r1_train, epoch_f1_1_train = [], [], [] # For 1-hop
            epoch_p_final_train, epoch_r_final_train, epoch_f1_final_train = [], [], [] # For final preds

            accumulated_loss_for_step = torch.tensor(0.0, device=self.device)
            train_pbar = tqdm(train_data_loader, desc=f"Epoch {ep+1} Training")
            batch_idx_in_epoch = 0

            for batch in train_pbar:
                if batch is None: continue 
                
                batch_avg_loss, final_preds_str, hop1_preds_str = self.forward_per_batch(batch)
                
                # ## MODIFIED: 分別計算1-hop和最終(可能是混合了1-hop和2-hop)的準確率
                p1, r1, f1_1 = self.measure_accuracy(hop1_preds_str, batch['hop1_target_cuis'])
                p_final, r_final, f1_final = self.measure_accuracy(final_preds_str, batch['hop2_target_cuis']) # 假設最終預測主要對標 hop2 GT

                # 記錄用於epoch平均的指標
                epoch_p1_train.append(p1); epoch_r1_train.append(r1); epoch_f1_1_train.append(f1_1)
                epoch_p_final_train.append(p_final); epoch_r_final_train.append(r_final); epoch_f1_final_train.append(f1_final)

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
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    train_pbar.set_postfix({'L': f'{accumulated_loss_for_step.item():.3f}', 
                                            'F1@1': f'{f1_1:.3f}', 'F1@Final': f'{f1_final:.3f}'})
                    accumulated_loss_for_step = torch.tensor(0.0, device=self.device)
                batch_idx_in_epoch +=1

            if batch_idx_in_epoch % update_step != 0 and accumulated_loss_for_step.item() > 0 :
                 torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0); torch.nn.utils.clip_grad_norm_(self.gmodel.parameters(), 1.0)
                 self.optimizer.step(); self.optimizer.zero_grad()

            avg_ep_train_loss = np.mean(epoch_loss_train_list) if epoch_loss_train_list else float('nan')
            avg_ep_f1_1_train = np.mean(epoch_f1_1_train) if epoch_f1_1_train else float('nan')
            avg_ep_f1_final_train = np.mean(epoch_f1_final_train) if epoch_f1_final_train else float('nan')
            print(f"\nEpoch {ep+1} Train Avg: Loss={avg_ep_train_loss:.4f}, F1@1={avg_ep_f1_1_train:.4f}, F1@Final={avg_ep_f1_final_train:.4f}")

            avg_ep_dev_loss, avg_ep_p1_dev, avg_ep_r1_dev, avg_ep_f1_1_dev, \
            avg_ep_p_final_dev, avg_ep_r_final_dev, avg_ep_f1_final_dev = self.validate(dev_data_loader)
            print(f"Epoch {ep+1} Valid Avg: Loss={avg_ep_dev_loss:.4f}, P@1={avg_ep_p1_dev:.4f}, R@1={avg_ep_r1_dev:.4f}, F1@1={avg_ep_f1_1_dev:.4f} | P@Final={avg_ep_p_final_dev:.4f}, R@Final={avg_ep_r_final_dev:.4f}, F1@Final={avg_ep_f1_final_dev:.4f}")

            if lr_scheduler: lr_scheduler.step()

            # ## MODIFIED: Early Stopping and Model Saving based on a primary metric, e.g., F1@Final
            current_metric_val = avg_ep_f1_final_dev # 或者 val_loss: avg_ep_dev_loss
            # ... (提前停止與模型保存邏輯，與之前類似，只是選擇比較的指標可能改變) ...
            # ... (例如，如果 F1@Final 提升則保存模型) ...
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
        epoch_p_final_dev, epoch_r_final_dev, epoch_f1_final_dev = [], [], []
        
        dev_pbar = tqdm(dev_data_loader, desc="Validation")
        with torch.no_grad():
            for batch in dev_pbar:
                if batch is None: continue
                batch_avg_loss, final_preds_str, hop1_preds_str = self.forward_per_batch(batch)
                
                p1, r1, f1_1 = self.measure_accuracy(hop1_preds_str, batch['hop1_target_cuis'])
                p_final, r_final, f1_final = self.measure_accuracy(final_preds_str, batch['hop2_target_cuis']) # 假設

                epoch_loss_dev_list.append(batch_avg_loss.item())
                epoch_p1_dev.append(p1); epoch_r1_dev.append(r1); epoch_f1_1_dev.append(f1_1)
                epoch_p_final_dev.append(p_final); epoch_r_final_dev.append(r_final); epoch_f1_final_dev.append(f1_final)
                dev_pbar.set_postfix({'L': f'{batch_avg_loss.item():.3f}', 'F1@1': f'{f1_1:.3f}', 'F1@Final': f'{f1_final:.3f}'})

        avg_loss = np.mean(epoch_loss_dev_list) if epoch_loss_dev_list else float('nan')
        avg_p1 = np.mean(epoch_p1_dev) if epoch_p1_dev else 0.0
        avg_r1 = np.mean(epoch_r1_dev) if epoch_r1_dev else 0.0
        avg_f1_1 = np.mean(epoch_f1_1_dev) if epoch_f1_1_dev else 0.0
        avg_p_final = np.mean(epoch_p_final_dev) if epoch_p_final_dev else 0.0
        avg_r_final = np.mean(epoch_r_final_dev) if epoch_r_final_dev else 0.0
        avg_f1_final = np.mean(epoch_f1_final_dev) if epoch_f1_final_dev else 0.0
        
        return avg_loss, avg_p1, avg_r1, avg_f1_1, avg_p_final, avg_r_final, avg_f1_final

# ====================== Main Block ======================
if __name__ =='__main__':

    

    # --- 常規設置與資源加載 (假設這些已存在或從 args 讀取) ---
    # !!! 確保這些變量已定義並加載好 !!!
    # args = parser.parse_args() # 如果使用 argparse
    TEST_TOKENIZER_PATH = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext" # 或者您的本地路徑
    GRAPH_NX_FILE = "./drknows/SNOMED_CUI_MAJID_Graph_wSelf.pkl" # 原始 NetworkX 圖
    
    CUI_EMBEDDING_FILE = "./drknows/GraphModel_SNOMED_CUI_Embedding.pkl"
    
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
        # 將 device 傳遞給 CuiEmbedding 的構造函數
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
    epochs = 1 
    LR = 1e-5 
    intermediate_loss_flag = True 
    contrastive_flag = True 
    batch_size = 4 
    
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
        hdim=hdim,
        nums_of_head=nums_of_head,
        cui_vocab_str_to_idx=cui_vocab_for_trainer,
        top_n=top_n,
        device=device,
        nums_of_epochs=epochs, 
        LR=LR,
        cui_weights_dict=None, 
        contrastive_learning=contrastive_flag,
        intermediate=intermediate_loss_flag,
        save_model_path=model_save_path,
        gnn_update=True, 
        path_encoder_type="Transformer",
        path_ranker_type="Flat",
        gnn_type="Stack", 
        gin_hidden_dim=gin_hidden_dim_val,
        gin_num_layers=gin_num_layers_val,
        #input_edge_dim_for_gin=actual_input_edge_dim_for_gin,
        early_stopping_patience=3,
        early_stopping_metric='val_loss'
    )
    print("Trainer instantiated.")

    print("\nCreating optimizer...")
    trainer_instance.create_optimizers()
    lr_scheduler_instance = None 

    print("\nLoading datasets...")
    try:
        train_dataset_obj = MediQAnnotatedDataset(TRAIN_ANNOTATION_FILE, tokenizer)
        dev_dataset_obj = MediQAnnotatedDataset(DEV_ANNOTATION_FILE, tokenizer)
    except Exception as e:
        print(f"Error loading datasets: {e}"); exit()
        
    if len(train_dataset_obj) == 0 or len(dev_dataset_obj) == 0:
        print("Error: A dataset is empty after loading!"); exit()

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

    
    

