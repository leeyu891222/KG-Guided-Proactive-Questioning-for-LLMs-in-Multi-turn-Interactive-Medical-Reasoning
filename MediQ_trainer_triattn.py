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
    """
    Loads annotated MediQ data (with facts, CUIs, and pre-calculated paths)
    and performs dynamic sampling of known/unknown facts for training.
    Derives hop-specific target CUIs for intermediate loss.
    """
    def __init__(self, annotation_file_path, tokenizer, random_seed=2023):
        """
        Args:
            annotation_file_path (str): Path to the JSON annotation file.
            tokenizer: Pre-initialized Hugging Face tokenizer (e.g., SapBERT).
            random_seed (int): Seed for reproducible random sampling.
        """
        print(f"載入標註數據: {annotation_file_path}")
        try:
            with open(annotation_file_path, 'r', encoding='utf-8') as f:
                self.all_annotations = json.load(f)
            # 過濾掉 facts 數量少於 2 的案例 (因為至少需要1知1未知)
            self.case_ids = [
                cid for cid, data in self.all_annotations.items()
                if len(data.get("atomic_facts", [])) >= 2
            ]
            if len(self.case_ids) < len(self.all_annotations):
                print(f"警告: 已過濾掉 {len(self.all_annotations) - len(self.case_ids)} 個 atomic_facts 少於 2 的案例。")
            print(f"成功載入並篩選後共 {len(self.case_ids)} 筆有效案例。")
        except FileNotFoundError:
            print(f"錯誤：找不到標註文件 {annotation_file_path}。")
            raise
        except Exception as e:
            print(f"載入標註文件時發生錯誤: {e}")
            raise

        self.tokenizer = tokenizer
        self.random = random.Random(random_seed) # Use an instance for reproducibility

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, index):
        """
        Fetches data for one case, performs dynamic sampling, prepares model inputs,
        and derives hop-specific target CUIs.
        """
        case_id = self.case_ids[index]
        data = self.all_annotations[case_id]

        atomic_facts = data["atomic_facts"]
        facts_cuis = data["facts_cuis"] # List of lists
        paths_between_facts = data.get("paths_between_facts", {}) # Use .get for safety
        num_facts = len(atomic_facts)

        # 之前的檢查已在 __init__ 中完成，這裡 num_facts 必定 >= 2

        # --- Dynamic Sampling ---
        all_indices = set(range(num_facts))
        known_indices = {0}
        remaining_indices = list(all_indices - known_indices)

        # 至少保留一個未知 fact
        num_unknown = self.random.randint(1, len(remaining_indices))
        unknown_indices = set(self.random.sample(remaining_indices, num_unknown))
        known_indices.update(set(remaining_indices) - unknown_indices)
        target_unknown_idx = self.random.choice(list(unknown_indices))
        # --- End Dynamic Sampling ---

        # --- Prepare Model Inputs ---
        # 1. Known Text Input (for h_text)
        known_texts = " ".join([atomic_facts[i] for i in sorted(list(known_indices))])
        input_text_tks = self.tokenizer(known_texts,
                                        truncation=True,
                                        padding="max_length",
                                        max_length=512, # Adjust as needed
                                        return_tensors="pt")
        input_text_tks = {k: v.squeeze(0) for k, v in input_text_tks.items()}

        # 2. Known CUIs (for h_con - returned as list for now)
        known_cuis_list = []
        for i in known_indices:
            # 確保 facts_cuis[i] 是列表，即使為空
            known_cuis_list.extend(facts_cuis[i] if isinstance(facts_cuis[i], list) else [])
        known_cuis_list = list(set(known_cuis_list)) # Unique known CUIs

        # 3. Candidate Paths (Retrieved based on known -> target_unknown)
        #    These are the pre-calculated GT paths for loss calculation reference
        gt_candidate_paths = []
        for i in known_indices:
            path_key = f"{i}_{target_unknown_idx}"
            if path_key in paths_between_facts:
                gt_candidate_paths.extend(paths_between_facts[path_key])

        # 4. Derive Hop-Specific Target CUIs from GT Paths
        hop1_target_cuis = set()
        hop2_target_cuis = set() # This will collect targets from both 1-hop and 2-hop paths

        for path in gt_candidate_paths:
            if not path: continue # Skip empty paths if any

            # Assuming path format [CUI_start, Rel1, CUI_mid, Rel2, CUI_end] (len 5) for 2-hop
            # or [CUI_start, Rel1, CUI_end] (len 3) for 1-hop
            path_len = len(path)
            if path_len == 3: # 1-hop path
                hop1_target_cuis.add(path[2]) # Add the target CUI
                hop2_target_cuis.add(path[2]) # Also add to final targets
            elif path_len == 5: # 2-hop path
                 # The intermediate node path[2] is NOT a hop-1 TARGET, it's a hop-1 step.
                 # Only the final node path[4] is added to hop2 targets.
                 # If intermediate loss needs targets reachable *after* 1 hop *via any path*,
                 # we might need a different definition or path structure.
                 # Current definition: hop1_target_cuis are CUIs reachable via exactly 1-hop paths.
                 hop2_target_cuis.add(path[4]) # Add the final target CUI
            # else: handle potential malformed paths?

        return {
            "case_id": case_id,
            "known_indices": sorted(list(known_indices)),
            "target_unknown_idx": target_unknown_idx,
            "input_text_tks": input_text_tks,
            "known_cuis": known_cuis_list,
            # "candidate_paths": gt_candidate_paths, # Keep if needed for triplet loss path embeddings
            "hop1_target_cuis": list(hop1_target_cuis), # Unique CUIs reachable in exactly 1 hop via GT paths
            "hop2_target_cuis": list(hop2_target_cuis)  # Unique CUIs reachable in 1 or 2 hops via GT paths
        }


# --- 修改後的 Collate Function ---
def collate_fn_mediq_paths(batch):
    """
    Collate function for the MediQAnnotatedDataset.
    Handles padding for tokenized text, returns lists for variable-length items.
    Filters out None items resulting from sampling errors in __getitem__.
    """
    # Filter out None items first
    batch = [item for item in batch if item is not None]
    if not batch: # If all items in the batch were None
        return None

    # Gather different parts of the data
    case_ids = [item['case_id'] for item in batch]
    known_indices = [item['known_indices'] for item in batch]
    target_unknown_idx = [item['target_unknown_idx'] for item in batch]
    input_ids = [item['input_text_tks']['input_ids'] for item in batch]
    attention_mask = [item['input_text_tks']['attention_mask'] for item in batch]
    known_cuis = [item['known_cuis'] for item in batch] # List of lists
    # candidate_paths = [item['candidate_paths'] for item in batch] # List of lists of lists
    hop1_target_cuis = [item['hop1_target_cuis'] for item in batch] # List of lists
    hop2_target_cuis = [item['hop2_target_cuis'] for item in batch] # List of lists


    # Pad tokenized text inputs
    # Assuming tokenizer.pad_token_id is 0, adjust if necessary
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    # Package the batch
    return {
        "case_id": case_ids,
        "known_indices": known_indices,
        "target_unknown_idx": target_unknown_idx,
        "input_text_tks_padded": {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded
        },
        "known_cuis": known_cuis, # Return as list of lists
        # "candidate_paths": candidate_paths, # Return if needed later
        "hop1_target_cuis": hop1_target_cuis, # Return as list of lists
        "hop2_target_cuis": hop2_target_cuis # Return as list of lists
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
    """
    Retrieves 1-hop neighbor paths using tensor operations.
    Args:
        current_cui_str_list (list[str]): List of CUI strings to start searching from.
        tensor_graph (dict): Dictionary containing tensorized graph representations
                             (adj_src, adj_tgt, adj_edge_type, cui_to_idx, etc.).
        prev_candidate_tensors (dict, optional): Tensors from the previous hop,
                                                 containing 'src_indices', 'tgt_indices'.
                                                 Used to build multi-hop path memory.
    Returns:
        tuple:
            - candidate_src_indices (torch.Tensor): Source CUI indices for found 1-hop paths. [num_paths]
            - candidate_tgt_indices (torch.Tensor): Target CUI indices for found 1-hop paths. [num_paths]
            - candidate_edge_type_indices (torch.Tensor): Edge type indices for found 1-hop paths. [num_paths]
            - path_memory_src_prev_hop (torch.Tensor or None): For 2+ hops, CUI indices of the *original* source
                                                               of the paths extended in this hop. [num_paths]
                                                               None for 1st hop.
    """
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

    # Find edges originating from the current_cui_indices_tensor
    # Create a mask for relevant source nodes in the graph's adjacency list
    # graph_adj_src shape: [total_num_edges_in_graph]
    # current_cui_indices_tensor shape: [num_current_sources]
    # We want to find all edges where graph_adj_src is one of current_cui_indices_tensor
    
    # Efficiently find matching edges:
    # Create a boolean mask by checking for each edge if its source is in current_cui_indices_tensor
    # This can be slow if current_cui_indices_tensor is large.
    # A better way for many sources: iterate through current_cui_indices_tensor and gather.
    
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

        if prev_candidate_tensors is not None and 'src_indices_orig' in prev_candidate_tensors:
            # If it's the second hop or more, we need to carry forward the *original* source CUI index
            # from the path that led to `src_idx_current_hop`.
            # `prev_candidate_tensors['tgt_indices']` contains the nodes that became sources for this hop.
            # `prev_candidate_tensors['src_indices_orig']` contains the original sources for those paths.
            
            # Find which of the previous hop's target nodes matches src_idx_current_hop
            prev_tgt_mask = (prev_candidate_tensors['tgt_indices'] == src_idx_current_hop)
            if torch.any(prev_tgt_mask):
                # Get the original source(s) that led to src_idx_current_hop
                # If multiple paths led to src_idx_current_hop, we might need to duplicate or handle this.
                # For simplicity, let's assume one main path or take the first.
                original_source_for_this_src = prev_candidate_tensors['src_indices_orig'][prev_tgt_mask][0] # Take the first one
                path_memory_src_prev_hop_list.append(original_source_for_this_src.repeat(srcs_for_paths.size(0)))
            else:
                # This case should be rare if src_idx_current_hop came from prev_candidate_tensors['tgt_indices']
                # If src_idx_current_hop was an initial CUI, and it's not the first hop, something is off.
                # For first hop, prev_candidate_tensors is None.
                # Fallback: use src_idx_current_hop if original source cannot be traced
                path_memory_src_prev_hop_list.append(src_idx_current_hop.repeat(srcs_for_paths.size(0)))
        else: # First hop, the source of the path is the source of the memory
            path_memory_src_prev_hop_list.append(src_idx_current_hop.repeat(srcs_for_paths.size(0)))


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
                 input_edge_dim_for_gin=108
                 ):
        super(GraphModel, self).__init__()
        self.g_tensorized = preprocess_graph_to_tensors(g_nx) # Preprocess NX graph
        self.n_encoder_lookup = cui_embedding_lookup # For string CUI to embedding
        
        # Edge encoder needs the tensorized graph's edge_to_idx for dynamic mapping
        # Or, if EdgeOneHot is modified to take graph_nx and build its own mapping:
        self.e_encoder = EdgeOneHot(graph=g_nx) # Assumes EdgeOneHot can handle NX graph
        # self.edge_idx_map = self.g_tensorized['edge_to_idx'] # No longer needed if e_encoder handles it

        self.p_encoder_type = path_encoder_type
        self.path_ranker_type = path_ranker_type
        
        edge_dim = self.e_encoder.onehot_mat.shape[-1]

        if self.p_encoder_type == "Transformer":
            self.p_encoder = PathEncoderTransformer(hdim, hdim + edge_dim)
        else:
            self.p_encoder = PathEncoder(hdim, hdim + edge_dim)

        if self.path_ranker_type == "Combo":
            self.p_ranker = TriAttnCombPathRanker(hdim)
        else:
            self.p_ranker = TriAttnFlatPathRanker(hdim)

        self.k_hops = num_hops
        self.path_per_batch_size = 128
        self.top_n = top_n
        self.cui_weights_dict = cui_weights_dict if cui_weights_dict else {}
        self.hdim = hdim # Store hdim for use
        self.device = device
        self.gnn_update = gnn_update
        self.gnn_type = gnn_type
        self.gin_num_layers = gin_num_layers if gin_num_layers else (3 if gnn_type == "Stack" else 1)
        self.gin_hidden_dim = gin_hidden_dim if gin_hidden_dim else hdim
        self.input_edge_dim_for_gin = input_edge_dim_for_gin # Should match e_encoder output
        if self.gnn_update:
            if self.gnn_type == "Stack":
                # GINStack input_node_dim is hdim, output_dim_final is hdim
                self.gnn = GINStack(
                    input_node_dim=hdim,
                    input_edge_dim=self.input_edge_dim_for_gin, # From e_encoder
                    hidden_dim=self.gin_hidden_dim,
                    output_dim_final=hdim, # GIN output should be hdim to match other embs
                    num_layers=self.gin_num_layers,
                    device=device
                )
            else: # Single GIN layer
                self.gnn = NodeAggregateGIN(
                    input_node_dim=hdim,
                    input_edge_dim=self.input_edge_dim_for_gin,
                    hidden_dim=self.gin_hidden_dim,
                    output_dim=hdim, # Output should be hdim
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

        # Get embeddings for these unique CUIs
        # These are the base embeddings before GIN (if any)
        unique_src_embs_base = self._get_embeddings_by_indices(unique_hop_src_indices)
        unique_tgt_embs = self._get_embeddings_by_indices(unique_hop_tgt_indices)

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
            # Prepare inputs for GIN:
            # x_src_unique: current_path_src_embs_for_encoding (features of unique source nodes)
            # unique_src_to_process_indices: unique_hop_src_indices (global indices of these source nodes)
            # path_source_indices_global_scatter: scatter_src_index_for_gin_agg (local indices for scatter)
            # path_target_node_features: Features of target nodes for *all paths*
            #    Need to get these from unique_tgt_embs based on cand_tgt_idx_hop
            map_global_tgt_idx_to_local_in_unique = {
                 glob_idx.item(): i for i, glob_idx in enumerate(unique_hop_tgt_indices)
            }
            gin_path_tgt_node_features = unique_tgt_embs[
                torch.tensor([map_global_tgt_idx_to_local_in_unique[glob_idx.item()] for glob_idx in cand_tgt_idx_hop],
                             dtype=torch.long, device=self.device)
            ]
            # path_edge_features: path_edge_embs_for_gin_and_path_enc

            # Ensure edge features have the dimension expected by GIN
            if path_edge_embs_for_gin_and_path_enc.shape[-1] != self.input_edge_dim_for_gin:
                print(f"Warning: Edge feature dim mismatch for GIN. Got {path_edge_embs_for_gin_and_path_enc.shape[-1]}, expected {self.input_edge_dim_for_gin}. Adjusting...")
                # This might involve padding or truncation, or re-configuring GIN's input_edge_dim
                # For now, let's try to use it as is if GIN's edge_transform_linear can handle it,
                # or error out, or use a subset/zeroes. This needs careful handling.
                # A simple fix if GIN expects specific dim is to ensure e_encoder produces that.
                # For this test, we'll assume it matches or GIN handles it.
                # If they must match, then self.e_encoder.onehot_mat.shape[-1] must be input_edge_dim_for_gin.
                # This is now set during GraphModel init.

            updated_src_embs_from_gin, _ = self.gnn(
                current_path_src_embs_for_encoding, # x_src_unique
                unique_hop_src_indices,             # unique_src_to_process_indices (global IDs)
                scatter_src_index_for_gin_agg,      # path_source_indices_global_scatter (local IDs for scatter)
                gin_path_tgt_node_features,         # path_target_node_features
                path_edge_embs_for_gin_and_path_enc # path_edge_features
            )
            current_path_src_embs_for_encoding = updated_src_embs_from_gin
            # print(f"GIN updated src embs shape: {current_path_src_embs_for_encoding.shape}")

        # --- 4. Prepare inputs for PathEncoder ---
        # We need one source embedding and one (target+edge) embedding for each path in cand_src_idx_hop
        
        # --- PathEncoder Input Prep ---
        path_specific_src_embs = current_path_src_embs_for_encoding[
            torch.tensor([map_global_src_idx_to_local_in_unique[idx.item()] for idx in cand_src_idx_hop],
                         dtype=torch.long, device=self.device)
        ]
        
        map_global_tgt_idx_to_local_in_unique = { # Re-define or ensure scope
             glob_idx.item(): i for i, glob_idx in enumerate(unique_hop_tgt_indices)
        }
        path_specific_tgt_embs = unique_tgt_embs[
            torch.tensor([map_global_tgt_idx_to_local_in_unique[idx.item()] for idx in cand_tgt_idx_hop],
                         dtype=torch.long, device=self.device)
        ]
        
        # Combined (target + edge) embeddings for PathEncoder input
        # path_edge_embs is already [num_paths, edge_dim]
        # path_specific_tgt_embs is [num_paths, hdim]
        # Ensure dimensions are broadcastable or correctly concatenated.
        # PathEncoder expects tgt_input_dim = hdim + edge_dim.
        try:
            combined_tgt_edge_embs_for_path_enc = torch.cat(
                (path_specific_tgt_embs, path_edge_embs_for_gin_and_path_enc), dim=-1
            )
        except RuntimeError as e:
            print(f"Error PathEnc concat: {e}, tgt_shape: {path_specific_tgt_embs.shape}, edge_shape: {path_edge_embs_for_gin_and_path_enc.shape}")
            return None, {}, None, None, True


        # --- 5. Path Encoding and Ranking ---
        num_paths_this_hop = cand_src_idx_hop.size(0)
        all_path_scores_hop, all_encoded_paths_hop = [], []

        for i in range(0, num_paths_this_hop, self.path_per_batch_size):
            s_ = slice(i, min(i + self.path_per_batch_size, num_paths_this_hop))
            src_b, combined_tgt_edge_b = path_specific_src_embs[s_], combined_tgt_edge_embs_for_path_enc[s_]
            
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


class Trainer(nn.Module):
    def __init__(self, tokenizer,
                 encoder, # Base encoder like SapBERT
                 g_nx, # NetworkX graph FOR GraphModel's preprocessing
                 cui_embedding_lookup, # Initialized CuiEmbedding object (string lookup)
                 hdim,
                 nums_of_head, # For original PathRanker, may not be used by TriAttn
                 cui_vocab_str_to_idx, # IMPORTANT: CUI string to GLOBAL CUI INDEX mapping
                 top_n,
                 device,
                 nums_of_epochs,
                 LR,
                 cui_weights_dict=None, # Dict: CUI_str -> weight
                 contrastive_learning=True,
                 save_model_path=None,
                 gnn_update=True,
                 intermediate=False, # Calculate loss on intermediate hops?
                 distance_metric="Cosine",
                 path_encoder_type="MLP",
                 path_ranker_type="Flat",
                 gnn_type="Stack",
                 gin_hidden_dim=None, # Pass to GraphModel
                 gin_num_layers=1,    # Pass to GraphModel
                 input_edge_dim_for_gin=108, # Pass to GraphModel
                 triplet_margin=1.0,
                 early_stopping_patience=3,
                 early_stopping_metric='val_loss',
                 early_stopping_delta=0.001
                 ):
        super(Trainer, self).__init__()

        self.tokenizer = tokenizer
        self.encoder = encoder
        self.k_hops = 2 # Max hops fixed at 2
        self.device = device
        self.cui_vocab_str_to_idx = cui_vocab_str_to_idx # Store this mapping
        self.rev_cui_vocab_idx_to_str = {v: k for k, v in cui_vocab_str_to_idx.items()}


        self.gmodel = GraphModel(
            g_nx=g_nx, # Pass the NetworkX graph
            cui_embedding_lookup=cui_embedding_lookup,
            hdim=hdim,
            nums_of_head=nums_of_head,
            num_hops=self.k_hops,
            top_n=top_n,
            device=device,
            cui_weights_dict=cui_weights_dict,
            gnn_update=gnn_update,
            path_encoder_type=path_encoder_type,
            path_ranker_type=path_ranker_type,
            gnn_type=gnn_type,
            gin_hidden_dim=gin_hidden_dim,
            gin_num_layers=gin_num_layers,
            input_edge_dim_for_gin=input_edge_dim_for_gin
        )

        self.encoder.to(device)
        self.gmodel.to(device) # GraphModel's __init__ should also move its submodules

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

        self.loss_fn_bce = nn.BCEWithLogitsLoss()
        self.save_model_path = save_model_path

        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric.lower()
        self.early_stopping_delta = early_stopping_delta
        self.epochs_no_improve = 0
        self.early_stop = False
        if self.early_stopping_metric == 'val_loss':
            self.best_metric_val = float('inf')
        elif self.early_stopping_metric == 'val_acc':
            self.best_metric_val = float('-inf')
        else:
            raise ValueError("early_stopping_metric 必須是 'val_loss' 或 'val_acc'")

        print("**** ============= TRAINER (MediQ Tensorized GModel) ============= **** ")
        # ... (exp_setting logging) ...
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
        known_cuis_str_batch = batch['known_cuis'] # List of lists of CUI strings
        hop1_target_cuis_str_batch = batch['hop1_target_cuis']
        hop2_target_cuis_str_batch = batch['hop2_target_cuis']

        input_task_embs_batch = self.encoder(
            input_text_tks_padded['input_ids'].to(self.device),
            input_text_tks_padded['attention_mask'].to(self.device)
        ).pooler_output # [batch_size, hdim]

        accumulated_batch_loss = torch.tensor(0.0, device=self.device)
        batch_size = input_task_embs_batch.shape[0]
        final_predicted_cuis_global_idx_batch = [torch.empty(0, dtype=torch.long, device=self.device) for _ in range(batch_size)]


        for i in range(batch_size):
            sample_loss_this_item = torch.tensor(0.0, device=self.device)
            task_emb_sample = input_task_embs_batch[i].unsqueeze(0) # [1, hdim]
            known_cuis_str_sample = known_cuis_str_batch[i]
            
            context_emb_sample = self.compute_context_embedding(known_cuis_str_sample)
            if context_emb_sample is None: # Fallback if no known CUIs have embeddings
                context_emb_sample = task_emb_sample # Use task embedding as context

            # Initial state for multi-hop for this sample
            current_cui_str_list_for_hop = known_cuis_str_sample
            prev_iter_state_for_next_hop = None # For the first hop
            
            # Store selected target CUIs (global indices) after the last successful hop for this sample
            last_hop_selected_target_cuis_idx = torch.empty(0, dtype=torch.long, device=self.device)


            for running_k in range(self.k_hops): # 0, 1 for k_hops=2
                if not current_cui_str_list_for_hop and running_k > 0 : # Stop if no nodes to expand from previous hop
                    break
                
                all_paths_info_hop, visited_next_hop_str_dict_hop, \
                next_hop_state_info_hop, stop_flag = self.gmodel.one_iteration(
                    task_emb_batch=task_emb_sample,
                    current_cui_str_list=current_cui_str_list_for_hop,
                    running_k_hop=running_k,
                    context_emb_batch=context_emb_sample,
                    prev_iteration_state=prev_iter_state_for_next_hop
                )

                if stop_flag or all_paths_info_hop is None:
                    print(f"  Sample {i}, Hop {running_k}: Stopped or no path info.")
                    break # Stop iteration for this sample if no paths or error

                # Determine Ground Truth CUI strings for this hop
                gt_cuis_str_list_this_hop = hop1_target_cuis_str_batch[i] if running_k == 0 else hop2_target_cuis_str_batch[i]

                # Calculate Loss for this hop
                hop_bce_loss = self.compute_bce_loss_for_hop(all_paths_info_hop, gt_cuis_str_list_this_hop)
                hop_triplet_loss = torch.tensor(0.0, device=self.device)
                if self.contrastive_learning and self.mode == "train":
                    # Construct anchor: e.g., combination of task and context
                    # For simplicity, let's use task_emb_sample as anchor.
                    # A better anchor might involve embeddings of the source CUIs of this hop.
                    anchor_for_triplet = task_emb_sample 
                    hop_triplet_loss = self.compute_triplet_loss_for_hop(
                        anchor_for_triplet, all_paths_info_hop, gt_cuis_str_list_this_hop
                    )
                
                current_hop_total_loss = hop_bce_loss + hop_triplet_loss
                
                #---debug---
                print(f"  Sample {i}, Hop {running_k}:")
                print(f"    Start CUIs: {current_cui_str_list_for_hop[:5] if current_cui_str_list_for_hop else 'None'}")
                print(f"    GT CUIs: {gt_cuis_str_list_this_hop[:5] if gt_cuis_str_list_this_hop else 'None'}")
                if all_paths_info_hop:
                    print(f"    Path Scores numel: {all_paths_info_hop['scores'].numel() if all_paths_info_hop['scores'] is not None else 'None'}")
                    print(f"    Path Tgt Idx numel: {all_paths_info_hop['tgt_idx'].numel() if all_paths_info_hop['tgt_idx'] is not None else 'None'}")
                else:
                    print(f"    all_paths_info_hop is None.")
                print(f"    BCE Loss: {hop_bce_loss.item():.4f}, Triplet Loss: {hop_triplet_loss.item():.4f}")
                print(f"    Hop Total Loss: {current_hop_total_loss.item():.4f}, Requires Grad: {current_hop_total_loss.requires_grad}, Grad Fn: {current_hop_total_loss.grad_fn}")
                #---debug---
                
                

                if running_k == 0 and self.intermediate:
                    sample_loss_this_item = sample_loss_this_item + current_hop_total_loss
                if running_k == self.k_hops - 1: # Always include final hop loss
                    sample_loss_this_item = sample_loss_this_item + current_hop_total_loss
                
                # Update state for the next iteration
                if next_hop_state_info_hop and next_hop_state_info_hop['selected_hop_target_idx'] is not None:
                    last_hop_selected_target_cuis_idx = next_hop_state_info_hop['selected_hop_target_idx']
                    current_cui_str_list_for_hop = [
                        self.rev_cui_vocab_idx_to_str.get(idx.item()) 
                        for idx in last_hop_selected_target_cuis_idx
                        if self.rev_cui_vocab_idx_to_str.get(idx.item()) is not None # Filter out Nones
                    ]
                    prev_iter_state_for_next_hop = next_hop_state_info_hop 
                else: # No valid targets selected for next hop
                    break 
            
            accumulated_batch_loss = accumulated_batch_loss + sample_loss_this_item
            final_predicted_cuis_global_idx_batch[i] = last_hop_selected_target_cuis_idx


        avg_batch_loss = accumulated_batch_loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=self.device)
        
        # Convert final predicted global indices back to CUI strings for measure_accuracy
        final_predicted_cuis_str_batch = []
        for idx_tensor in final_predicted_cuis_global_idx_batch:
            final_predicted_cuis_str_batch.append(
                [self.rev_cui_vocab_idx_to_str.get(idx.item()) for idx in idx_tensor 
                 if self.rev_cui_vocab_idx_to_str.get(idx.item()) is not None]
            )
            
        return avg_batch_loss, final_predicted_cuis_str_batch # Return CUI strings for accuracy


    def measure_accuracy(self, final_predicted_cuis_str_batch, target_cuis_str_batch, mode="Recall@N"):
        # (measure_accuracy logic using string lists, as defined before)
        # ...
        batch_size = len(target_cuis_str_batch)
        if batch_size == 0: return 0.0
        accs = []
        for i in range(batch_size):
            gold_cuis = set(target_cuis_str_batch[i])
            pred_cuis = set(final_predicted_cuis_str_batch[i]) # Already list of strings
            num_pred = len(pred_cuis)
            num_gold = len(gold_cuis)
            num_intersect = len(gold_cuis.intersection(pred_cuis))

            precision = num_intersect / num_pred if num_pred > 0 else 0.0
            recall = num_intersect / num_gold if num_gold > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Defaulting to recall for this setup, adjust 'mode' as needed
            acc = recall 
            accs.append(acc)
        return np.mean(accs) if accs else 0.0


    def train(self, train_data_loader, dev_data_loader, lr_scheduler=None):
        # (Training loop logic, as defined before, calling self.forward_per_batch)
        # Ensure self.encoder and self.gmodel are set to train() mode at start of epoch
        # and eval() mode for validation.
        # ...
        if self.optimizer is None: self.create_optimizers()
        update_step = 4 

        for ep in range(self.nums_of_epochs):
            print(f"\n--- Starting Epoch {ep+1}/{self.nums_of_epochs} ---")
            self.mode = 'train'
            self.encoder.train()
            self.gmodel.train()
            epoch_loss_train_list = [] # Store individual batch losses
            epoch_acc_train_list = []
            
            # For gradient accumulation
            accumulated_loss_for_step = torch.tensor(0.0, device=self.device)
            
            train_pbar = tqdm(train_data_loader, desc=f"Epoch {ep+1} Training")
            batch_idx_in_epoch = 0

            for batch in train_pbar:
                if batch is None: continue # From collate_fn

                # forward_per_batch returns average loss for that batch
                batch_avg_loss, final_predictions_str = self.forward_per_batch(batch)
                
                # Debug Start
                
                if not batch_avg_loss.requires_grad:
                    print(f"----------- ALERT: Epoch {ep+1}, Batch {batch_idx_in_epoch+1} -----------")
                    print(f"batch_avg_loss: {batch_avg_loss.item()}, requires_grad: {batch_avg_loss.requires_grad}, grad_fn: {batch_avg_loss.grad_fn}")
                    print("This batch resulted in a loss that does not require gradients.")
                    print("This usually means no valid paths/targets led to a parameter-dependent loss calculation for ANY sample in this batch.")
                    
                    # 在這種情況下，調用 .backward() 會出錯。
                    # 我們可以選擇跳過這個 micro-batch 的梯度更新。
                    # 仍然需要處理梯度累積的計數器 batch_idx_in_epoch % update_step
                    if (batch_idx_in_epoch + 1) % update_step == 0:
                        # 如果剛好是更新步驟，但這個 micro-batch (或之前累積的) 沒有梯度
                        # 我們可能仍然需要 zero_grad() 來清除任何可能的舊梯度（儘管理論上不應該有）
                        # 並且不執行 optimizer.step() 如果沒有有效的梯度累積。
                        # 一個簡單的處理是，如果整個 update_step 週期內的 loss 都是 non-grad，則不 step。
                        # 但目前的梯度累積是每個 micro-batch 都 .backward()。
                        # 所以，如果 loss_to_accumulate 不 require_grad，就不 backward。
                        print("Skipping backward pass for this micro-batch as loss does not require grad.")
                        # accumulated_loss_for_step 應該不會被這個 non-grad loss 影響，因為我們只加 loss.item()
                    # else: # 不是 optimizer step 的邊界，繼續累積（雖然這個 batch 貢獻的梯度為0）
                    #     pass

                
                
                
                
                # Debug End
                
                # For accuracy, use hop2_target_cuis (final hop GT)
                target_cuis_str_for_acc = batch['hop2_target_cuis']
                batch_acc = self.measure_accuracy(final_predictions_str, target_cuis_str_for_acc)

                if torch.isnan(batch_avg_loss) or torch.isinf(batch_avg_loss):
                    print(f"Warning: NaN/Inf loss detected at Epoch {ep+1}, Batch {batch_idx_in_epoch+1}. Skipping step.")
                    # Important: if skipping, zero out any accumulated grad for THIS batch
                    self.optimizer.zero_grad() 
                    accumulated_loss_for_step = torch.tensor(0.0, device=self.device) # Reset for next accumulation cycle
                    batch_idx_in_epoch +=1
                    continue
                
                # Normalize loss for accumulation
                loss_to_accumulate = batch_avg_loss / update_step
                loss_to_accumulate.backward()
                accumulated_loss_for_step = accumulated_loss_for_step + loss_to_accumulate.detach() # Track accumulated loss for logging

                epoch_loss_train_list.append(batch_avg_loss.item()) # Log actual (non-normalized) batch average loss
                epoch_acc_train_list.append(batch_acc)

                if (batch_idx_in_epoch + 1) % update_step == 0:
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.gmodel.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Log accumulated loss
                    avg_loss_over_update_steps = accumulated_loss_for_step.item() # This is sum of normalized losses
                    train_pbar.set_postfix({'Step Loss': f'{avg_loss_over_update_steps:.4f}', 'Batch Acc': f'{batch_acc:.4f}'})
                    accumulated_loss_for_step = torch.tensor(0.0, device=self.device) # Reset for next cycle
                
                batch_idx_in_epoch +=1

            # Handle any remaining gradients if epoch size not multiple of update_step
            if batch_idx_in_epoch % update_step != 0 and accumulated_loss_for_step > 0 :
                 torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                 torch.nn.utils.clip_grad_norm_(self.gmodel.parameters(), 1.0)
                 self.optimizer.step()
                 self.optimizer.zero_grad()

            avg_epoch_train_loss = np.mean(epoch_loss_train_list) if epoch_loss_train_list else float('nan')
            avg_epoch_train_acc = np.mean(epoch_acc_train_list) if epoch_acc_train_list else float('nan')
            print(f"\nEpoch {ep+1} Avg Training Loss: {avg_epoch_train_loss:.4f}, Avg Training Acc: {avg_epoch_train_acc:.4f}")

            # Validation, LR Scheduling, Early Stopping (as before)
            # ...
            avg_epoch_dev_loss, avg_epoch_dev_acc = self.validate(dev_data_loader) # Pass loader
            print(f"Epoch {ep+1} Validation Loss: {avg_epoch_dev_loss:.4f}, Validation Acc: {avg_epoch_dev_acc:.4f}")

            if lr_scheduler:
                lr_scheduler.step() # Or lr_scheduler.step(avg_epoch_dev_loss) if ReduceLROnPlateau

            # Early Stopping Logic (as before)
            current_metric_val = avg_epoch_dev_loss if self.early_stopping_metric == 'val_loss' else avg_epoch_dev_acc
            improved = False
            if self.early_stopping_metric == 'val_loss':
                if current_metric_val < self.best_metric_val - self.early_stopping_delta:
                    improved = True
            else: # val_acc
                if current_metric_val > self.best_metric_val + self.early_stopping_delta:
                    improved = True
            
            if improved:
                print(f"Metric improved ({self.best_metric_val:.4f} --> {current_metric_val:.4f}). Saving model...")
                self.best_metric_val = current_metric_val
                self.epochs_no_improve = 0
                if self.save_model_path:
                    try:
                        torch.save(self.gmodel.state_dict(), self.save_model_path)
                        encoder_save_path = os.path.join(os.path.dirname(self.save_model_path), "encoder.pth")
                        torch.save(self.encoder.state_dict(), encoder_save_path)
                        print(f"Model saved to {self.save_model_path} and {encoder_save_path}")
                    except Exception as e: print(f"Error saving model: {e}")
            else:
                self.epochs_no_improve += 1
                print(f"Metric did not improve for {self.epochs_no_improve} epoch(s). Best: {self.best_metric_val:.4f}")

            if self.epochs_no_improve >= self.early_stopping_patience:
                self.early_stop = True
                print(f"\nEarly stopping triggered after {ep+1} epochs.")
                break 
            print("-" * 50)
        
        if not self.early_stop: print("Training finished after all epochs.")


    def validate(self, dev_data_loader):
        # (Validation loop logic, as defined before, calling self.forward_per_batch)
        # ...
        print("Running validation...")
        self.mode = 'eval' # Set mode to eval
        self.encoder.eval()
        self.gmodel.eval()
        epoch_loss_dev_list = []
        epoch_acc_dev_list = []
        dev_pbar = tqdm(dev_data_loader, desc="Validation")

        with torch.no_grad():
            for batch in dev_pbar:
                if batch is None: continue
                batch_avg_loss, final_predictions_str = self.forward_per_batch(batch)
                target_cuis_str_for_acc = batch['hop2_target_cuis'] # Final hop GT
                batch_acc = self.measure_accuracy(final_predictions_str, target_cuis_str_for_acc)

                epoch_loss_dev_list.append(batch_avg_loss.item())
                epoch_acc_dev_list.append(batch_acc)
                dev_pbar.set_postfix({'Loss': f'{batch_avg_loss.item():.4f}', 'Acc': f'{batch_acc:.4f}'})

        avg_loss = np.mean(epoch_loss_dev_list) if epoch_loss_dev_list else float('nan')
        avg_acc = np.mean(epoch_acc_dev_list) if epoch_acc_dev_list else float('nan')
        return avg_loss, avg_acc


# ====================== Main Block ======================
if __name__ =='__main__':

   
    
    # --- 常規設置與資源加載 (假設這些已存在或從 args 讀取) ---
    # !!! 確保這些變量已定義並加載好 !!!
    # args = parser.parse_args() # 如果使用 argparse
    TEST_TOKENIZER_PATH = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext" # 或者您的本地路徑
    GRAPH_NX_FILE = "./drknows/SNOMED_CUI_MAJID_Graph_wSelf.pkl" # 原始 NetworkX 圖
    
    CUI_EMBEDDING_FILE = "./drknows/GraphModel_SNOMED_CUI_Embedding.pkl"
    
    TRAIN_ANNOTATION_FILE = "./MediQ/mediq_train_annotations.json"
    DEV_ANNOTATION_FILE = './MediQ/mediq_dev_annotations.json'
    
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
        print("真實 CuiEmbedding 實例化成功。")
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
    epochs = 300 
    LR = 1e-5 
    intermediate_loss_flag = True 
    contrastive_flag = True 
    batch_size = 1 
    
    gin_hidden_dim_val = hdim 
    gin_num_layers_val = 2  


    temp_edge_enc = EdgeOneHot(graph=g_nx_loaded)
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
        cui_embedding_lookup=cui_embedding_lookup_obj, # *** 使用真實的嵌入對象 ***
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
        path_encoder_type="MLP",
        path_ranker_type="Flat",
        gnn_type="Stack", 
        gin_hidden_dim=gin_hidden_dim_val,
        gin_num_layers=gin_num_layers_val,
        input_edge_dim_for_gin=actual_input_edge_dim_for_gin,
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

    train_loader_instance = DataLoader(train_dataset_obj, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_mediq_paths)
    dev_loader_instance = DataLoader(dev_dataset_obj, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_mediq_paths)
    print("Dataloaders created.")

    print("\n" + "="*30 + "\n STARTING Tensorized Trainer RUN (REAL EMBEDDINGS) \n" + "="*30 + "\n")
    try: 
        trainer_instance.train(train_loader_instance, dev_loader_instance, lr_scheduler_instance)
    except Exception as e:
        print(f"ERROR DURING TRAINING RUN: {e}")
        import traceback
        traceback.print_exc()
    print("\n" + "="*30 + "\n TENSORIZED TRAINER RUN (REAL EMBEDDINGS) FINISHED \n" + "="*30 + "\n")


    

