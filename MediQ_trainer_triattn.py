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



def retrieve_neighbors_paths_no_self(cui_lists, g, prev_candidate_paths_df):
    #import queue 
    """Important function to reformat paths and direct neighbors 
    Input: 
    cui_lists: a list of CUIs that start searching
    g: current graph 
    candidate_paths_df: if not None, it is the history one-hot path from previous traversal iteration
    Output:
    all_paths: a list of one-hot hop given cui_lists 
    all_neighbors: a list of concepts that will be the candidate predictions
    path_memories: a list of list with four elements: visited source nodes from the prev iteration; 
                                                      starting nodes at the current iteration (aka cui_list);
                                                      current candidate node, 
                                                      current candidate edge 
    """
    cui_neighbors = retrieve_subgraphs(cui_lists, g) # dictionary of cuis and their neighrbos 
    all_neighbors = [] 
    all_paths = [] 
    path_memories = [] # dict or list? 
    path_buffer = {} # path buffer, a list of dictionary indicating what sources lead to the current target
    if prev_candidate_paths_df is None: 
        all_neighbors = [vv[0] for k,v in cui_neighbors.items() for vv in v if len(v) !=0] # list of neighbor nodes 
        all_paths = [[k, vv[0], vv[1]] for k,v in cui_neighbors.items() for vv in v if len(v) !=0] # list of one-hop path
        path_memories = [[[k], k, vv[0], vv[1]] for k,v in cui_neighbors.items() for vv in v if len(v) !=0]
    else:
        # faster version using itertuples 
        for _ in  prev_candidate_paths_df.itertuples():
            src, tgt = _.Src, _.Tgt
            if src == tgt:
                continue 
            if tgt in path_buffer:
                path_buffer[tgt].append(src)
            else:
                path_buffer[tgt]= [src]
        # remove a specific path where it is the self edge at the first hop 
        for k,v in cui_neighbors.items():
            #print("path buffer k {path_buffer[k]} given k", )
            if len(v) == 0:
                continue
            if k not in path_buffer:
                record = [k,k,k,"self"]
                if record not in path_memories:
                    path_memories.append(record)
                continue 
            for vv in v:
                path_memories.append([path_buffer[k], k, vv[0], vv[1]]) 

        all_paths =[pm[1:] for pm in path_memories] 
        all_neighbors =[pm[-2] for pm in path_memories]
    #print("ALL NEIGHRBORS", all_neighbors)
    #print("PATH MEMO: ", path_memories)
    return all_paths, all_neighbors, path_memories  



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
    Backpropagated NOT required, dictionary look-up layer  
    This module could be used for CUI embedding (loaded from pre-trained SAPBERT vectors)
    Module has been tested
    Need to rewrite to read from existing embeddings 
    """
    def __init__(self, embedding_file):
        super(CuiEmbedding, self).__init__() 
        self.data = pickle.load(open(embedding_file, 'rb')) 

    def encode(self, cui_lists):
        outputs = [] 
        outputs = [torch.as_tensor(self.data[c]) for c in cui_lists] 
        return torch.stack(outputs).squeeze(1)

    def update(self, cui_idx_dicts, cui_embeddings):
        for _, c in enumerate(cui_idx_dicts): # c: CUI, v: index in embedding lookup layer 
            self.data[c] = cui_embeddings[_,:].unsqueeze(0).detach().cpu() # something like this 


class EdgeOneHot(object):
    """
    Dynamically builds edge mapping from the graph and provides one-hot embeddings.
    """
    def __init__(self, graph: nx.DiGraph, unknown_rel_label: str = "UNKNOWN_REL"):
        """
        Initializes by extracting unique edge labels from the graph.

        Args:
            graph (nx.DiGraph): The NetworkX graph object.
            unknown_rel_label (str): Label to use for edges without a 'label' attribute
                                     or for labels encountered during lookup but not seen during init.
        """
        super().__init__()
        print("動態生成 Edge Mappings from Graph...")
        unique_edge_labels = set()
        edge_count = 0
        missing_label_count = 0
        for u, v, data in graph.edges(data=True):
            edge_count += 1
            label = data.get('label') # Safely get the label attribute
            if label is not None:
                unique_edge_labels.add(label)
            else:
                missing_label_count += 1
                unique_edge_labels.add(unknown_rel_label) # Add default label if missing

        if missing_label_count > 0:
             print(f"警告: 圖譜中有 {missing_label_count}/{edge_count} 條邊缺少 'label' 屬性。已使用 '{unknown_rel_label}' 代替。")

        # Create mapping from label to index
        self.edge_mappings = {label: i for i, label in enumerate(sorted(list(unique_edge_labels)))}
        self.num_edge_types = len(self.edge_mappings)
        self.unknown_rel_index = self.edge_mappings.get(unknown_rel_label) # Store index for unknown/missing

        print(f"完成 Edge Mappings 生成。共找到 {self.num_edge_types} 種唯一邊緣類型。")
        # print(f"Edge Mappings: {self.edge_mappings}") # Uncomment for debugging

        # Create one-hot matrix
        self.onehot_mat = F.one_hot(torch.arange(0, self.num_edge_types), num_classes=self.num_edge_types).float() # Use float for embeddings

    def Lookup(self, edge_labels: list):
        """
        Looks up one-hot vectors for a list of edge labels.

        Args:
            edge_labels (list): List of edge label strings.

        Returns:
            torch.Tensor: Tensor containing one-hot embeddings for the labels.
        """
        indices = []
        for e in edge_labels:
            if e in self.edge_mappings:
                indices.append(self.edge_mappings[e])
            elif self.unknown_rel_index is not None:
                 print(f"警告: Lookup 時遇到未知邊緣標籤 '{e}'。使用 UNKNOWN_REL 索引。")
                 indices.append(self.unknown_rel_index)
            else:
                 # Should not happen if unknown_rel_label was added during init, but as a fallback:
                 print(f"警告: Lookup 時遇到未知邊緣標籤 '{e}' 且無 UNKNOWN_REL 索引。使用索引 0。")
                 indices.append(0) # Fallback to index 0

        indices_tensor = torch.tensor(indices, dtype=torch.long)
        # Ensure indices are within bounds before lookup
        indices_tensor = torch.clamp(indices_tensor, 0, self.num_edge_types - 1)
        vectors = self.onehot_mat[indices_tensor]

        return vectors


class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.mlp_layer1 = nn.Linear(input_dim, hidden_dim, bias=False) 
        self.mlp_layer2 = nn.Linear(hidden_dim, output_dim, bias=False) 
        nn.init.xavier_uniform_(self.mlp_layer1.weight)
        nn.init.xavier_uniform_(self.mlp_layer2.weight)
        self.linears.append(self.mlp_layer1)
        self.linears.append(self.mlp_layer2)

        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        if h.shape[0] == 1: # if only one example, dont use batchnorm 
            h = F.relu(self.linears[0](h), inplace=True) 
        else:
            h = F.relu(self.batch_norm(self.linears[0](h)),inplace=True)
        #print("MLP layer batch norm: ", h)
        return self.linears[1](h)
    
class GINStack(nn.Module):
    """
    Stacking NodeAggregateGIN
    """
    def __init__(self, dim_h, device):
        super().__init__()
        self.conv1 = NodeAggregateGIN(dim_h, dim_h, dim_h, device)
        self.conv2 = NodeAggregateGIN(dim_h, dim_h, dim_h, device)
        self.conv3 = NodeAggregateGIN(dim_h, dim_h, dim_h, device)
        self.lin1 = nn.Linear(dim_h*3, dim_h)
        self.lin2 = nn.Linear(dim_h, dim_h)

    def forward(self, paths_srcs, path_tgt_edges_per_src, candidate_paths_df):
        h1, src_dicts1 = self.conv1(paths_srcs, path_tgt_edges_per_src, candidate_paths_df)
        h2, src_dicts2 = self.conv2(paths_srcs, path_tgt_edges_per_src, candidate_paths_df)
        h3, src_dicts3 = self.conv3(paths_srcs, path_tgt_edges_per_src, candidate_paths_df)

        h = torch.cat((h1,h2,h3), dim=1)

        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5)
        h= self.lin2(h)
        self.src_df_dicts = self.conv1.src_df_dicts 

        return h, src_dicts1 

class NodeAggregateGIN(nn.Module):
    """On-the-fly neighboring aggregation for candidate nodes
    Source: MLP-Based Graph Isomorphism (Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (ICLR 2018). 
    "How powerful are graph neural networks?")
    Graph Isomorphism Network with Edge Features, introduced by
    `Strategies for Pre-training Graph Neural Networks <https://arxiv.org/abs/1905.12265>
    h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \sum_{j\in\mathcal{N}(i)}\mathrm{ReLU}(h_j^{l} + e_{j,i}^{l})\right)
    where :math:`e_{j,i}^{l}` is the edge feature.

    """
    def __init__(self, input_dim, hidden_dim, output_dim, device, init_eps=0, learn_eps=False):
        super().__init__()
        self.edge_linear = nn.Linear(hidden_dim+108,hidden_dim)
        self.aggr = MLP(input_dim, hidden_dim, output_dim) 
        self.device = device 
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([init_eps]))


    def message(self, path_tgt_edges_per_src, edge_dicts):
        # \sum_{j\in\mathcal{N}(i)}\mathrm{ReLU}(h_j^{l} + e_{j,i}^{l})\right) 
        msgs = F.relu(path_tgt_edges_per_src).to('cuda')  # same dimensionality as h_n (node embedding, which is 768)
        msgs_dict = {}
        for k,v in edge_dicts.items():
            indices = torch.tensor(v).to(self.device)
            indices = indices.to(torch.device('cuda'))
            # print("indices: ", indices)
            # raise NotImplementedError
            msgs_dict[k] = torch.sum(msgs[indices], dim=0).unsqueeze(0)

        return msgs_dict 

    def organize_neighbors(self, candidate_paths_df):
        r"""Return two dictionaries to help organize the paths and embeddings:
        outputs: dictionary where key is the source node, values are the neighboring nodes and edges 
        src_dicts: dictionary where key is the source node, values is the (start) index of the source node in the df  
        """
        outputs = {}
        src_dicts = OrderedDict()
        # convert df (all paths) to dict structure where key is the source node, values are the neighboring nodes and edges 
        for rowid, item in candidate_paths_df.iterrows():
            src = item['Src']
            if src in outputs:
                outputs[src].append(rowid)
            else:
                outputs[src] = [rowid]
                src_dicts[src] = [rowid] 
        return outputs, src_dicts  

    def forward(self, paths_srcs, path_tgt_edges_per_src, candidate_paths_df):
        #output = self.aggr(node_repr)   
        df_edge_dicts, src_dicts = self.organize_neighbors(candidate_paths_df)
        cand_cuis_mappings = {k: v for v, k in enumerate(set(candidate_paths_df['Src'].to_list()))}
        sorted_cand_cuis_mappings_keys = sorted(list(cand_cuis_mappings.keys()))
        for v,k in enumerate(sorted_cand_cuis_mappings_keys):
            cand_cuis_mappings[k] = v  

        self.src_dicts = src_dicts # for debugging purporse: {CUI: index in path dataframe}
        self.src_df_dicts = df_edge_dicts # to compute CL
        # updated msg
        msgs_dict = self.message(path_tgt_edges_per_src, df_edge_dicts)
        outputs = []
        new_src_dicts = {} 
        count = 0
        for k,v in src_dicts.items():
            new_src_dicts[k] = count
            count += 1
            h_src = paths_srcs[cand_cuis_mappings[k]].unsqueeze(0)
            #h_src = paths_srcs[torch.tensor(v[0]).to(self.device)] # h_src original embedding
            h_msg = self.edge_linear(msgs_dict[k]) 
            h_n_prime = (1 + self.eps) * h_src + h_msg 
            outputs.append(h_n_prime)
        raw_feats = torch.cat(outputs) 
        #print(f"Raw Features {raw_feats.shape}")
        output = self.aggr(raw_feats.squeeze(1))
        return output, new_src_dicts  


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
                 g,
                 cui_embedding, # cui_objects
                 hdim,
                 nums_of_head, # Note: May not be used by TriAttn rankers
                 # edges_dicts, # edge name to index mapping for EdgeOneHot
                 # cui_aui_mappings, # Original mapping, potentially unused now
                 nums_of_hops, # Should be fixed to 2 based on prior discussion
                 top_n,
                 device,
                 cui_weights=None,
                 gnn_update=True, # Use GIN?
                 cui_flag=True, # Used by original PathRanker, maybe not TriAttn
                 path_encoder_type="MLP", # "MLP" or "Transformer"
                 path_ranker_type="Flat", # "Flat" or "Combo"
                 gnn_type="Stack", # "Stack" or "Single"
                 prune_thsh=0.8): # Original prune threshold, maybe unused now
        super(GraphModel, self).__init__()
        self.n_encoder = cui_embedding # CuiEmbedding object
        self.e_encoder = EdgeOneHot(graph=g) # EdgeOneHot object
        self.edges_mappings = self.e_encoder.edge_mappings
        self.p_encoder_type = path_encoder_type
        self.path_ranker_type = path_ranker_type

        # Instantiate Path Encoder based on type
        # Note: Original encoders expect specific inputs (src_emb, tgt+edge_emb).
        # If encoding pre-calculated paths differently, this might need adjustment later.
        if self.p_encoder_type == "Transformer":
            # Transformer version expects edge dim + node dim = path_dim
            edge_dim = self.e_encoder.onehot_mat.shape[-1] if self.e_encoder else 0
            self.p_encoder = PathEncoderTransformer(hdim, hdim + edge_dim)
        else: # Default to MLP version
            edge_dim = self.e_encoder.onehot_mat.shape[-1] if self.e_encoder else 0
            self.p_encoder = PathEncoder(hdim, hdim + edge_dim)

        # Instantiate Path Ranker based on type
        if self.path_ranker_type == "Combo":
            self.p_ranker = TriAttnCombPathRanker(hdim)
        else: # Default to Flat version
            self.p_ranker = TriAttnFlatPathRanker(hdim)

        self.g = g # networkx graph object
        self.k = nums_of_hops # max k hops (should be 2)
        self.path_per_batch_size = 128 # Batch size for processing paths internally
        self.top_n = top_n
        # self.logit_loss_mode = "last" # Original param, maybe remove if loss handled in Trainer
        self.cui_weights = cui_weights # Dictionary mapping CUI -> weight

        # self.edges_mappings = edges_dicts # Keep for e_encoder
        # self.visited_paths = {} # Tracks paths to selected CUIs (CUI -> path string)
        # self.src_cui_emb = {} # Tracks embeddings used to reach selected CUIs (CUI -> embedding tensor)
        self.device = device
        # self.candidate_paths_df = None # Stores DataFrame of paths found in the current hop
        self.gnn_update = gnn_update
        # self.prune_thsh = prune_thsh # Store if prune_paths util is used
        self.gnn_type = gnn_type

        # Instantiate GNN based on type if gnn_update is True
        if self.gnn_update:
            if self.gnn_type == "Stack":
                self.gnn = GINStack(hdim, self.device)
            else: # Default to single layer GIN
                self.gnn = NodeAggregateGIN(hdim, hdim, hdim, self.device)
        else:
             self.gnn = None # No GNN if update is false

    def one_iteration(self, task_emb, cui_lists, running_k, context_emb=None, prev_candidate_paths_df=None, prev_visited_paths_str=None):
        """
        執行一輪迭代：查找鄰居 -> 更新/獲取源嵌入 -> 編碼路徑 -> 評分 -> 選擇TopN
        Args:
            task_emb: 當前任務/文本嵌入
            cui_lists: 當前迭代的起始 CUI 列表
            running_k: 當前是第幾跳 (0 or 1 for k=2)
            context_emb: 上下文 CUI 嵌入
            prev_candidate_paths_df: 上一輪的 candidate_paths_df (用於 retrieve_neighbors_paths_no_self)
            prev_visited_paths_str: 上一輪選出的 visited_paths (CUI -> path string 字典)
        Returns:
            final_scores: 當前跳找到的路徑的分數 Tensor
            visited_paths_next_hop: 選出的 TopN CUIs 及其對應路徑字符串 (字典)
            candidate_paths_df: 當前跳找到的所有路徑的 DataFrame
            visited_path_embs_tensor: 當前跳所有路徑的嵌入 Tensor (for Triplet Loss)
            stop_flag: 是否提前停止
        """
        stop_flag = False # Reset stop flag for this iteration

        # 1. Dynamic neighbor finding using the utility function
        candidate_paths, candidate_neighbors, path_memories = retrieve_neighbors_paths_no_self(
            cui_lists, self.g, prev_candidate_paths_df # 使用上一輪的 df
        )
        '''
        # +++ Debugging: Check raw candidate_paths +++
        print(f"\n--- Debug Info (Inside one_iteration, k={running_k}, Start) ---")
        print(f"Input cui_lists (first 10): {cui_lists[:10]}")
        print(f"Number of candidate_paths from retrieve_neighbors: {len(candidate_paths)}")
        malformed_paths_found = False
        for i, p in enumerate(candidate_paths):
             # More robust check: is it a list of 3 strings?
             if not (isinstance(p, list) and len(p) == 3 and all(isinstance(item, str) for item in p)):
                print(f"  !!! Malformed path found at index {i}: Type={type(p)}, Value={p}")
                malformed_paths_found = True
        if not malformed_paths_found:
             print("  All candidate_paths seem correctly formatted (list of 3 strings).")
        else:
             print("  !!! Found malformed paths in candidate_paths list !!!")
             # Decide how to handle: filter or return error? Let's filter for now
             original_count = len(candidate_paths)
             candidate_paths = [p for p in candidate_paths if isinstance(p, list) and len(p) == 3 and all(isinstance(item, str) for item in p)]
             print(f"  Filtered malformed paths. Count reduced from {original_count} to {len(candidate_paths)}.")
        # +++ End Check +++
        '''
        if not candidate_paths:
            print(f"  No valid candidate paths found after filtering or initially.")
            return None, {}, pd.DataFrame(columns=['Src', 'Tgt', 'Edge']), None, True

        # --- 2. Prepare Embeddings ---
        try:
            candidate_paths_df = pd.DataFrame(candidate_paths, columns=['Src', 'Tgt', 'Edge'])
            # print(f"  Successfully created candidate_paths_df, shape: {candidate_paths_df.shape}")
            # path_mem_df is needed only if using Transformer PathEncoder with multi-source memory
            # path_mem_df = pd.DataFrame(path_memories, columns=['Prev','Src', 'Tgt', 'Edge'])
        except ValueError as e:
            print(f"!!! Error creating DataFrame from candidate_paths: {e} !!!")
            print(f"Problematic candidate_paths list (first 10 entries): {candidate_paths[:10]}")
            return None, {}, pd.DataFrame(columns=['Src', 'Tgt', 'Edge']), None, True
        
        unique_neighbors = list(set(candidate_neighbors))
        unique_srcs = list(set(candidate_paths_df['Src'].tolist())) # Current hop's source nodes

        if not unique_neighbors or not unique_srcs:
            return None, {}, candidate_paths_df, None, True

        cand_neighbors_mappings = {cui: i for i, cui in enumerate(unique_neighbors)}
        cand_src_mappings = {cui: i for i, cui in enumerate(unique_srcs)}

        try:
            # Always get base embeddings for neighbors and sources using n_encoder
            cand_neighbors_vs = self.n_encoder.encode(unique_neighbors).to(self.device)
            cand_src_vs_base = self.n_encoder.encode(unique_srcs).to(self.device) # Base embeddings, no grad needed yet
        except KeyError as e:
            print(f"Error encoding CUI at hop {running_k}: {e}. Skipping iteration.")
            return None, {}, candidate_paths_df, None, True
        except Exception as e:
             print(f"Unexpected error encoding CUIs at hop {running_k}: {e}")
             return None, {}, candidate_paths_df, None, True

        # Apply CUI weights only at the first hop (k=0)
        if running_k == 0 and self.cui_weights:
             starting_cui_weights = torch.tensor([self.cui_weights.get(c, 1.0) for c in unique_srcs], device=self.device).unsqueeze(1)
             cand_src_vs = cand_src_vs_base * starting_cui_weights
        else:
             cand_src_vs = cand_src_vs_base # Use base embeddings for k > 0 or if no weights

        # Prepare edge embeddings and combined target+edge embeddings
        all_paths_tgt_edges_embs = []
        edge_mapping_keys = self.edges_mappings.keys() if self.e_encoder else set()
        for i in range(len(candidate_paths_df)):
            target_cui = candidate_paths_df.iloc[i]['Tgt']
            edge_label = candidate_paths_df.iloc[i]['Edge']

            # Get target CUI embedding
            tgt_emb = cand_neighbors_vs[cand_neighbors_mappings[target_cui]].unsqueeze(0)

            # Get edge embedding (use zero vector if no encoder or label unknown)
            if self.e_encoder and edge_label in edge_mapping_keys:
                edge_idx = self.edges_mappings[edge_label]
                # Ensure index is valid before lookup
                if 0 <= edge_idx < len(self.e_encoder.onehot_mat):
                    e_emb = self.e_encoder.onehot_mat[edge_idx].unsqueeze(0).float().to(self.device) # Ensure float and on device
                else:
                    print(f"Warning: Invalid edge index {edge_idx} for label {edge_label}. Using zero vector.")
                    e_emb = torch.zeros(1, self.e_encoder.onehot_mat.shape[-1], device=self.device) # Zero vector
            else:
                 # Get dimensionality if possible, otherwise use a default?
                 edge_dim = self.e_encoder.onehot_mat.shape[-1] if self.e_encoder else 1 # Use 1 if no encoder
                 e_emb = torch.zeros(1, edge_dim, device=self.device)

            # Combine target and edge embeddings
            # Ensure dimensions match PathEncoder expectation (hdim + edge_dim)
            if tgt_emb.shape[1] + e_emb.shape[1] != self.p_encoder.tgt_weights.in_features: # Check input dim of PathEncoder's target linear layer
                  print(f"Warning: Dimension mismatch for PathEncoder input. Tgt emb shape: {tgt_emb.shape}, Edge emb shape: {e_emb.shape}, Expected combined dim: {self.p_encoder.tgt_weights.in_features}")
                  # Fallback: maybe just use target embedding? or zero vector?
                  # For now, let's create a combined tensor trying to match dimensions, potentially with zeros
                  expected_dim = self.p_encoder.tgt_weights.in_features
                  combined_emb = torch.zeros(1, expected_dim, device=self.device)
                  len_tgt = min(expected_dim, tgt_emb.shape[1])
                  combined_emb[0, :len_tgt] = tgt_emb[0, :len_tgt]
                  if expected_dim > len_tgt and e_emb.shape[1] > 0:
                      len_edge = min(expected_dim - len_tgt, e_emb.shape[1])
                      combined_emb[0, len_tgt : len_tgt + len_edge] = e_emb[0, :len_edge]

                  # combined_emb = torch.cat((tgt_emb, e_emb), dim=-1) # Original simple concat
            else:
                 combined_emb = torch.cat((tgt_emb, e_emb), dim=-1)


            all_paths_tgt_edges_embs.append(combined_emb)

        if not all_paths_tgt_edges_embs:
             return None, {}, candidate_paths_df, None, True # Stop if no valid path components generated

        paths_tgt_edges = torch.cat(all_paths_tgt_edges_embs) # Shape [num_paths, hdim + edge_dim]

        # --- 3. Optional: Update Source Embeddings with GIN ---
        if self.gnn_update and self.gnn:
            try:
                # GIN requires source embeddings, combined target+edge embeddings per source, and the path df
                # Note: GIN's forward expects paths_srcs (cand_src_vs), path_tgt_edges_per_src (paths_tgt_edges), candidate_paths_df
                h_gnn_outputs, updated_node_dicts = self.gnn(cand_src_vs, paths_tgt_edges, candidate_paths_df)
                # Map GNN output back to the order needed for PathEncoder based on the path dataframe
                update_cui_idx_in_hgnn = updated_node_dicts # Assuming dict maps CUI -> index in h_gnn_outputs
                paths_srcs_for_encoder = torch.stack([
                    h_gnn_outputs[update_cui_idx_in_hgnn[candidate_paths_df.iloc[i]['Src']]]
                    for i in range(len(candidate_paths_df))
                    if candidate_paths_df.iloc[i]['Src'] in update_cui_idx_in_hgnn # Ensure Src exists
                ])
                # Adjust paths_tgt_edges to match filtered paths_srcs_for_encoder
                paths_tgt_edges = torch.stack([
                    all_paths_tgt_edges_embs[i].squeeze(0) # Need to manage dimensions here
                    for i in range(len(candidate_paths_df))
                     if candidate_paths_df.iloc[i]['Src'] in update_cui_idx_in_hgnn
                ])
                # Update the candidate_paths_df to only include rows where Src was updated by GNN
                valid_rows_idx = [i for i, row in candidate_paths_df.iterrows() if row['Src'] in update_cui_idx_in_hgnn]
                candidate_paths_df = candidate_paths_df.loc[valid_rows_idx].reset_index(drop=True)

            except Exception as e:
                 print(f"Error during GNN update: {e}")
                 # Fallback to using non-updated embeddings or stop? Let's fallback.
                 paths_srcs_for_encoder = torch.stack([
                    cand_src_vs[cand_src_mappings[candidate_paths_df.iloc[i]['Src']]]
                    for i in range(len(candidate_paths_df))
                 ])
                 paths_tgt_edges = torch.cat(all_paths_tgt_edges_embs) # Use original combined embs
        else: # No GNN update
            # Prepare source embeddings for PathEncoder - needs one embedding per path instance
             paths_srcs_for_encoder = torch.stack([
                cand_src_vs[cand_src_mappings[candidate_paths_df.iloc[i]['Src']]]
                for i in range(len(candidate_paths_df))
             ])
             paths_tgt_edges = torch.cat(all_paths_tgt_edges_embs) # Use original combined embs


        # --- 4. Path Encoding and Ranking ---
        B = paths_tgt_edges.shape[0]
        if B == 0:
             return None, {}, candidate_paths_df, None, True

        path_scores = []
        visited_path_embs_list = []
        all_indices = torch.arange(0, B, device=self.device)

        for i in range(0, B, self.path_per_batch_size):
            indices = all_indices[i : i + self.path_per_batch_size]
            src_embs_batch = paths_srcs_for_encoder[indices]
            path_tgt_edge_embs_batch = paths_tgt_edges[indices]

            # Encode paths -> path_h should require grad if p_encoder is trainable
            path_h = self.p_encoder(src_embs_batch, path_tgt_edge_embs_batch)
            visited_path_embs_list.append(path_h) # Store path embedding (requires_grad=True)

            # Rank paths -> scores should require grad if path_h or ranker params are trainable
            exp_task_emb = task_emb.expand(path_h.shape[0], -1)
            # Use context_emb (no grad) or task_emb (grad) for h_con input
            exp_context_emb_ranker = context_emb.expand(path_h.shape[0], -1) if context_emb is not None else exp_task_emb
            scores = self.p_ranker(exp_task_emb, exp_context_emb_ranker, path_h)
            path_scores.append(scores)

        final_scores = torch.cat(path_scores, dim=0) if path_scores else None
        visited_path_embs_tensor = torch.cat(visited_path_embs_list, dim=0) if visited_path_embs_list else None # Has grad history

        # --- 5. Top-N Selection (Pruning for next hop) ---
        if final_scores is None or final_scores.numel() == 0:
             return final_scores, {}, candidate_paths_df, visited_path_embs_tensor, True

        num_paths_found = final_scores.shape[0]
        current_top_n = min(self.top_n, num_paths_found)
        vals, pred_indices = torch.topk(final_scores, current_top_n, dim=0)
        valid_pred_indices_np = pred_indices.squeeze().cpu().numpy()

        # Get embeddings of the selected paths
        top_path_embs = visited_path_embs_tensor[valid_pred_indices_np] if visited_path_embs_tensor is not None else None

        # Select top candidate rows/row using the numpy indices
        # This might return a DataFrame OR a Series
        top_candidate_selection = candidate_paths_df.iloc[valid_pred_indices_np.tolist()]

        # --- 6. Prepare outputs for Trainer ---
        visited_paths_next_hop = {} # CUI -> path string dict for next iteration's state

        if top_path_embs is not None:
            # --- !!! 區分處理 Series (單行) 和 DataFrame (多行) !!! ---
            if isinstance(top_candidate_selection, pd.Series):
                # --- Handle Single Row (Series) Case ---
                row = top_candidate_selection
                # Embedding for this single path is top_path_embs (shape [1, hdim] or [hdim])
                # Ensure we get the single embedding tensor correctly
                single_path_emb = top_path_embs[0] if top_path_embs.shape[0] == 1 else top_path_embs

                try:
                    new_src = row['Tgt']
                    edge_label = row['Edge']
                    src_label = row['Src']

                    # Construct path string
                    path_str_prefix = prev_visited_paths_str.get(src_label, src_label) if prev_visited_paths_str else src_label
                    p = f"{path_str_prefix} --> {edge_label} --> {new_src}"
                    visited_paths_next_hop[new_src] = p
                    # No need to store embedding state for next hop

                except KeyError as e:
                    print(f"\n!!! KeyError processing single row Series !!!")
                    print(f"Content of Series 'row':\n{row}")
                    print(f"Error message: Missing key {e}")
                    # Decide how to handle, maybe stop? raise e
                except Exception as e:
                    print(f"\n!!! Unexpected Error processing single row Series !!!")
                    print(f"Error message: {e}")
                    # raise e

            elif isinstance(top_candidate_selection, pd.DataFrame):
                # --- Handle Multiple Rows (DataFrame) Case ---
                top_candidate_df = top_candidate_selection # Rename for clarity
                # Iterate using positional index (0 to current_top_n - 1)
                for positional_idx in range(len(top_candidate_df)):
                    try:
                        row = top_candidate_df.iloc[positional_idx] # Get row Series by position

                        new_src = row['Tgt']
                        edge_label = row['Edge']
                        src_label = row['Src']

                        # Construct path string
                        path_str_prefix = prev_visited_paths_str.get(src_label, src_label) if prev_visited_paths_str else src_label
                        p = f"{path_str_prefix} --> {edge_label} --> {new_src}"
                        visited_paths_next_hop[new_src] = p
                        # No need to store embedding state for next hop

                    except KeyError as e:
                       print(f"\n!!! KeyError caught inside loop !!!")
                       print(f"Positional Index where error occurred: {positional_idx}")
                       print(f"Content of 'row':\n{row}")
                       print(f"Error message: Missing key {e}")
                       raise e
                    except Exception as e:
                       print(f"\n!!! Unexpected Error caught inside loop !!!")
                       print(f"Positional Index where error occurred: {positional_idx}")
                       print(f"Error message: {e}")
                       raise e
            else:
                 # This case should ideally not happen if candidate_paths_df is valid
                 print(f"Warning: top_candidate_selection is neither Series nor DataFrame. Type: {type(top_candidate_selection)}")
            # --- 結束區分處理 ---
        else: # top_path_embs is None
            print("Warning: Cannot determine next hop info as top_path_embs is None.")

        # Return values needed by Trainer:
        # Note: candidate_paths_df here refers to the *original* one for this hop, used for loss calc
        return final_scores, visited_paths_next_hop, candidate_paths_df, visited_path_embs_tensor, stop_flag

class Trainer(nn.Module):
    def __init__(self, tokenizer,
                 encoder, # Base encoder like SapBERT
                 g, # NetworkX graph
                 cui_embedding, # Initialized CuiEmbedding object
                 hdim, # Hidden dimension (e.g., 768 for SapBERT)
                 nums_of_head, # Heads for original PathRanker (MultiheadAttention version) - may not be used by TriAttn
                 # all_edge_mappings, # Dict mapping edge labels to indices (for EdgeOneHot)
                 cui_vocab, # Dict mapping CUIs to vocab indices
                 # nums_of_hops, # Replaced by self.k = 2
                 top_n, # Top N paths to select per hop
                 device,
                 nums_of_epochs,
                 LR,
                 cui_weights=None, # Dict mapping CUIs to weights
                 contrastive_learning=True,
                 save_model_path=None,
                 # save_cui_embedding_path=None, # CuiEmbedding object passed in, maybe save externally
                 gnn_update=True, # Flag to enable/disable GIN updates within GraphModel
                 cui_flag=True, # Flag for original PathRanker - may not be used by TriAttn
                 intermediate=False, # Calculate loss on intermediate hops?
                 distance_metric="Cosine", # For triplet loss
                 path_encoder_type="MLP", # Or "Transformer"
                 path_ranker_type="Flat", # Or "Combo" (For TriAttn versions)
                 gnn_type="Stack", # Or "Single"
                 prune_thsh=0.8, # Threshold used in original prune_paths util
                 triplet_margin=1.0, # Margin for triplet loss
                 early_stopping_patience=3,    # 默認容忍3個epoch
                 early_stopping_metric='val_loss', # 監控驗證損失
                 early_stopping_delta=0.001    # 至少要有這麼多改善才算
                 ): 
        super(Trainer, self).__init__()

        self.tokenizer = tokenizer
        self.encoder = encoder # The base SapBERT model
        # self.CUI_encoder = cui_embedding # This is passed to GraphModel
        self.k = 2 # Max hops fixed at 2
        self.gmodel = GraphModel(g=g, 
                                 cui_embedding=cui_embedding,
                                 hdim=hdim,
                                 nums_of_head=nums_of_head,
                                 # edges_dicts removed
                                 nums_of_hops=self.k, # Pass self.k
                                 top_n=top_n,
                                 device=device,
                                 cui_weights=cui_weights,
                                 gnn_update=gnn_update,
                                 cui_flag=cui_flag,
                                 path_encoder_type=path_encoder_type,
                                 path_ranker_type=path_ranker_type,
                                 gnn_type=gnn_type,
                                 prune_thsh=prune_thsh)

        # Ensure models are on the correct device BEFORE optimizer creation
        self.encoder.to(device)
        self.gmodel.to(device)

        self.device = device
        self.LR = LR
        self.adam_epsilon = 1e-8
        self.weight_decay = 1e-4 # Example value, adjust as needed
        self.nums_of_epochs = nums_of_epochs
        self.intermediate = intermediate
        self.print_step = 50 # Print progress every N steps
        self.distance_metric = distance_metric
        # self.prune_thsh = prune_thsh # Already passed to GraphModel
        self.mode = 'train' # Default mode
        self.contrastive_learning = contrastive_learning
        self.triplet_margin = triplet_margin

        self.g = g # Keep graph reference if needed directly in Trainer
        self.loss_fn_bce = nn.BCEWithLogitsLoss()
        self.cui_vocab = cui_vocab # Needed for loss calculation against GT CUIs
        self.rev_cui_vocab = {v: k for k, v in self.cui_vocab.items()} # For converting vocab indices back to CUIs

        self.save_model_path = save_model_path
        # self.save_cui_embedding_path = save_cui_embedding_path # Handle CUI embedding saving outside if needed

        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric.lower() # val_loss or val_acc
        self.early_stopping_delta = early_stopping_delta
        self.epochs_no_improve = 0
        self.early_stop = False
        if self.early_stopping_metric == 'val_loss':
            self.best_metric_val = float('inf') # For loss, lower is better
        elif self.early_stopping_metric == 'val_acc':
            self.best_metric_val = float('-inf') # For accuracy, higher is better
        else:
            raise ValueError("early_stopping_metric 必須是 'val_loss' 或 'val_acc'")
        

        print("**** ============= **** ")
        exp_setting = f"TRAINER SETUP (MediQ Adapted): MAX HOPS: {self.k} | TOP N: {top_n} | INTERMEDIATE LOSS: {self.intermediate} | LR: {LR} | GNN UPDATE: {gnn_update} | CONTRASTIVE: {self.contrastive_learning} | PATH ENC: {path_encoder_type} | PATH RANKER: TriAttn {path_ranker_type} | GNN TYPE: {gnn_type}"
        print(exp_setting)
        if logging.getLogger().hasHandlers():
             logging.info(exp_setting)
        print("**** ============= **** ")

        self.optimizer = None # Initialize optimizer later

    def create_optimizers(self):
        """Creates AdamW optimizer for trainable parameters."""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.gmodel.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.gmodel.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # Use a single LR for now, original code had possibility of list but wasn't fully implemented
        effective_lr = self.LR[0] if isinstance(self.LR, list) else self.LR
        print(f"Using Learning Rate: {effective_lr}")
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=effective_lr, eps=self.adam_epsilon)
        print("Optimizer created.")

    def compute_context_embedding(self, known_cuis_list):
        """
        Computes the context embedding (h_con) from a list of known CUIs.
        Strategy: Average the embeddings of unique known CUIs.
        Args:
            known_cuis_list (list): List of CUI strings for one sample.
        Returns:
            torch.Tensor: Context embedding tensor (1 x hdim) or None if no valid CUIs.
        """
        if not known_cuis_list:
            # Return a zero tensor or handle appropriately
            # Returning None might be safer to indicate missing context
            return None
            # Or return torch.zeros(1, 768).to(self.device) # Assuming hdim=768

        # Ensure CUIs are valid and get embeddings
        valid_embeddings = []
        unique_known_cuis = list(set(known_cuis_list)) # Process unique CUIs
        for cui in unique_known_cuis:
            # Check if CUI exists in the embedding data (handle potential KeyError)
            if cui in self.gmodel.n_encoder.data:
                 # Assuming encode returns shape [1, hdim] or [hdim]
                 emb = self.gmodel.n_encoder.encode([cui]).to(self.device) # Ensure it's on device
                 valid_embeddings.append(emb.squeeze(0)) # Squeeze potential batch dim
            # else:
            #     print(f"Warning: CUI {cui} not found in embedding data.")

        if not valid_embeddings:
            # Return None if no known CUIs had embeddings
             return None
             # Or return torch.zeros(1, 768).to(self.device)

        # Aggregate embeddings (e.g., mean pooling)
        h_con = torch.mean(torch.stack(valid_embeddings), dim=0, keepdim=True) # Shape [1, hdim]
        return h_con

    def compute_bce_loss_for_hop(self, final_scores, candidate_paths_df, gt_cuis):
        """
        Calculates BCE loss for a given hop based on path scores.
        Aggregates scores for each target CUI and compares against GT.
        Args:
            final_scores (torch.Tensor): Tensor of scores for each path in candidate_paths_df.
            candidate_paths_df (pd.DataFrame): DataFrame with ['Src', 'Tgt', 'Edge'] for scored paths.
            gt_cuis (list): List of ground truth CUI strings for this hop.
        Returns:
            torch.Tensor: Calculated BCE loss for this hop.
        """
        hop_loss = torch.tensor(0.0).to(self.device)
        if final_scores is None or final_scores.numel() == 0 or not gt_cuis:
            return hop_loss # Return 0 loss if no scores or no GT

        # Map target CUIs from paths to scores
        target_cui_scores = defaultdict(list)
        if 'Tgt' not in candidate_paths_df.columns:
             print("Warning: 'Tgt' column missing in candidate_paths_df for BCE loss.")
             return hop_loss

        for i, score in enumerate(final_scores):
            target_cui = candidate_paths_df.iloc[i]['Tgt']
            target_cui_scores[target_cui].append(score)

        if not target_cui_scores:
             return hop_loss

        # Aggregate scores (e.g., max pooling) and calculate loss
        unique_target_cuis_in_paths = list(target_cui_scores.keys())
        gt_cuis_set = set(gt_cuis)

        aggregated_scores = []
        labels = []

        # Consider all unique target CUIs found in the paths for this hop
        for cui in unique_target_cuis_in_paths:
            # Aggregate scores for this CUI (e.g., max)
            max_score = torch.max(torch.stack(target_cui_scores[cui]))
            aggregated_scores.append(max_score)
            # Create label (1 if CUI is in GT, 0 otherwise)
            labels.append(1.0 if cui in gt_cuis_set else 0.0)

        if not aggregated_scores:
             return hop_loss

        # Calculate BCE loss
        scores_tensor = torch.stack(aggregated_scores).to(self.device)
        labels_tensor = torch.tensor(labels).to(self.device)
        hop_loss = self.loss_fn_bce(scores_tensor, labels_tensor)

        # print(f"Debug BCE: scores={scores_tensor.detach().cpu().numpy()}, labels={labels_tensor.cpu().numpy()}, loss={hop_loss.item()}")

        return hop_loss

    def compute_triplet_loss_for_hop(self, anchor_embedding, visited_path_embs, candidate_paths_df, gt_cuis):
        """
        Calculates Triplet loss for a given hop.
        Args:
            anchor_embedding (torch.Tensor): Anchor embedding (e.g., from h_text, h_con). Shape [1, hdim].
            visited_path_embs (torch.Tensor): Embeddings of the dynamically generated paths for this hop. Shape [num_paths, hdim].
            candidate_paths_df (pd.DataFrame): DataFrame mapping paths to target CUIs.
            gt_cuis (list): List of ground truth CUI strings for this hop.
        Returns:
            torch.Tensor: Calculated Triplet loss for this hop.
        """
        triplet_loss = torch.tensor(0.0).to(self.device)
        if anchor_embedding is None or visited_path_embs is None or visited_path_embs.numel() == 0 or not gt_cuis:
            return triplet_loss

        gt_cuis_set = set(gt_cuis)
        positive_path_indices = []
        negative_path_indices = []

        if 'Tgt' not in candidate_paths_df.columns:
            print("Warning: 'Tgt' column missing in candidate_paths_df for Triplet loss.")
            return triplet_loss

        for i in range(len(candidate_paths_df)):
            target_cui = candidate_paths_df.iloc[i]['Tgt']
            if target_cui in gt_cuis_set:
                positive_path_indices.append(i)
            else:
                negative_path_indices.append(i)

        # Only compute loss if we have at least one positive and one negative
        if not positive_path_indices or not negative_path_indices:
            return triplet_loss

        # Simple Triplet Loss: Compare each positive to all negatives (or a sample)
        # More sophisticated sampling might be needed for large numbers of negatives
        num_positives = len(positive_path_indices)
        num_negatives = len(negative_path_indices)

        # Expand anchor for broadcasting
        anchor_expanded = anchor_embedding.expand(num_positives * num_negatives, -1)

        # Get positive and negative embeddings
        positive_embs = visited_path_embs[positive_path_indices] # Shape [num_pos, hdim]
        negative_embs = visited_path_embs[negative_path_indices] # Shape [num_neg, hdim]

        # Repeat positive and negative embeddings to form all pairs
        # shape: [num_pos * num_neg, hdim]
        positive_embs_repeated = positive_embs.repeat_interleave(num_negatives, dim=0)
        # shape: [num_pos * num_neg, hdim]
        negative_embs_repeated = negative_embs.repeat(num_positives, 1)

        # Calculate distances (using Cosine Similarity or Pairwise Distance)
        if self.distance_metric == "Cosine":
            # Maximize similarity for positives, minimize for negatives
            # Loss = max(0, margin - cos(anchor, pos) + cos(anchor, neg))
            sim_pos = F.cosine_similarity(anchor_expanded, positive_embs_repeated)
            sim_neg = F.cosine_similarity(anchor_expanded, negative_embs_repeated)
            losses = F.relu(self.triplet_margin - sim_pos + sim_neg)
        else: # Euclidean distance
            # Minimize distance for positives, maximize for negatives
            # Loss = max(0, margin + dist(anchor, pos) - dist(anchor, neg))
            dist_pos = F.pairwise_distance(anchor_expanded, positive_embs_repeated, p=2)
            dist_neg = F.pairwise_distance(anchor_expanded, negative_embs_repeated, p=2)
            losses = F.relu(self.triplet_margin + dist_pos - dist_neg)

        triplet_loss = losses.mean()
        # print(f"Debug Triplet: pos_indices={len(positive_path_indices)}, neg_indices={len(negative_path_indices)}, loss={triplet_loss.item()}")

        return triplet_loss


    def forward_per_batch(self, batch):
        """
        Performs forward pass for a batch using the iterative GraphModel.
        Calculates loss based on hop-specific ground truths.
        Handles state (current CUIs, path strings) between iterations.
        """
        # 1. Unpack Batch & Get Initial Embeddings
        input_text_tks_padded = batch['input_text_tks_padded']
        known_cuis_batch = batch['known_cuis']
        hop1_target_cuis_batch = batch['hop1_target_cuis']
        hop2_target_cuis_batch = batch['hop2_target_cuis']

        input_task_embs = self.encoder(
            input_text_tks_padded['input_ids'].to(self.device),
            input_text_tks_padded['attention_mask'].to(self.device)
        ).pooler_output

        batch_total_loss = torch.tensor(0.0, requires_grad=True).to(self.device) # Ensure initial loss requires grad if accumulated upon
        batch_size = input_task_embs.shape[0]
        final_predicted_cuis_batch = [[] for _ in range(batch_size)]

        # 2. Iterate through samples in the batch
        for i in range(batch_size):
            # Initialize state for this sample
            sample_loss_accumulator = [] # Store loss tensors per hop
            task_emb = input_task_embs[i].unsqueeze(0)
            known_cuis_sample = known_cuis_batch[i]
            hop1_gt = hop1_target_cuis_batch[i]
            hop2_gt = hop2_target_cuis_batch[i]

            context_emb = self.compute_context_embedding(known_cuis_sample)
            if context_emb is None:
                context_emb = task_emb # Fallback

            # Initial state for iteration
            current_cui_list = known_cuis_sample
            current_candidate_paths_df = None # Previous hop's df
            current_visited_paths_str = None # Previous hop's path strings {CUI: path_str}

            # 3. Iterative Reasoning (Max K=2 hops)
            for running_k in range(self.k): # k = 0, 1
                if not current_cui_list: # Stop if no nodes to expand from previous hop
                    break

                final_scores, visited_paths_next_hop, candidate_paths_df_this_hop, \
                visited_path_embs_tensor, stop_flag = self.gmodel.one_iteration(
                    task_emb=task_emb,
                    cui_lists=current_cui_list,
                    running_k=running_k,
                    context_emb=context_emb,
                    prev_candidate_paths_df=current_candidate_paths_df, # Pass previous df
                    prev_visited_paths_str=current_visited_paths_str  # Pass previous path strings
                )

                if stop_flag or final_scores is None or final_scores.numel() == 0:
                    break # Stop if iteration signals stop or finds nothing

                # Determine Ground Truth for this hop
                current_gt_cuis = hop1_gt if running_k == 0 else hop2_gt

                # Calculate Loss for this hop (ensure loss tensors require grad)
                hop_bce_loss = self.compute_bce_loss_for_hop(final_scores, candidate_paths_df_this_hop, current_gt_cuis)
                hop_triplet_loss = torch.tensor(0.0).to(self.device)
                if self.contrastive_learning and self.mode == "train":
                    anchor_embedding = task_emb * context_emb # Simple anchor example
                    hop_triplet_loss = self.compute_triplet_loss_for_hop(anchor_embedding, visited_path_embs_tensor, candidate_paths_df_this_hop, current_gt_cuis)

                # Store loss tensor for accumulation
                if running_k == 0 and self.intermediate:
                    sample_loss_accumulator.append(hop_bce_loss + hop_triplet_loss)
                if running_k == self.k - 1: # Always include final hop loss
                    sample_loss_accumulator.append(hop_bce_loss + hop_triplet_loss)


                # Update state for the next iteration
                current_cui_list = list(visited_paths_next_hop.keys())
                current_candidate_paths_df = candidate_paths_df_this_hop # Store for next iteration's retrieve_neighbors...
                current_visited_paths_str = visited_paths_next_hop # Store path strings for next iteration


            # Store final predicted CUIs
            final_predicted_cuis_batch[i] = current_cui_list # CUIs selected after the last successful hop

            # Sum losses for the sample
            if sample_loss_accumulator:
                 batch_total_loss = batch_total_loss + torch.sum(torch.stack(sample_loss_accumulator))


        # 4. Return Average Batch Loss and Predictions
        avg_batch_loss = batch_total_loss / batch_size if batch_size > 0 else torch.tensor(0.0).to(self.device)
        final_predicted_cuis_batch_detached = [[cui for cui in sample_preds] for sample_preds in final_predicted_cuis_batch]
        return avg_batch_loss, final_predicted_cuis_batch_detached


    def measure_accuracy(self, final_predicted_cuis_batch, target_cuis_batch, mode="Recall@N"):
        """
        Measures accuracy based on comparing final predicted CUIs with hop-2 GT CUIs.
        Args:
            final_predicted_cuis_batch (list[list[str]]): List of predicted CUI lists for the batch.
            target_cuis_batch (list[list[str]]): List of hop-2 ground truth CUI lists.
            mode (str): Calculation mode ("Precision@N", "Recall@N", "F1@N"). N is determined by len(pred).
        Returns:
            float: Average accuracy score for the batch.
        """
        batch_size = len(target_cuis_batch)
        if batch_size == 0:
            return 0.0

        accs = []
        for i in range(batch_size):
            gold_cuis = set(target_cuis_batch[i])
            pred_cuis = set(final_predicted_cuis_batch[i])
            num_pred = len(pred_cuis)
            num_gold = len(gold_cuis)
            num_intersect = len(gold_cuis.intersection(pred_cuis))

            precision = num_intersect / num_pred if num_pred > 0 else 0.0
            recall = num_intersect / num_gold if num_gold > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            if mode == "Precision@N":
                acc = precision
            elif mode == "Recall@N":
                acc = recall
            elif mode == "F1@N":
                 acc = f1
            else: # Default to Recall
                acc = recall

            accs.append(acc)

        return np.mean(accs)


    def train(self, train_data, dev_data, lr_scheduler=None):
        """Main training loop with early stopping."""
        if self.optimizer is None:
            self.create_optimizers()

        # min_dev_loss = float('inf') # 由 self.best_metric_val 取代
        update_step = 4 # Gradient accumulation steps

        for ep in range(self.nums_of_epochs):
            print(f"\n--- Starting Epoch {ep+1}/{self.nums_of_epochs} ---")
            self.mode = 'train'
            self.encoder.train()
            self.gmodel.train()
            epoch_loss_train = []
            epoch_acc_train = []
            batch_count = 0
            accumulated_loss = 0.0
            train_pbar = tqdm(train_data, desc=f"Epoch {ep+1} Training")

            for batch in train_pbar:
                if batch is None: continue
                batch_loss, final_predictions = self.forward_per_batch(batch)
                target_cuis_batch = batch['hop2_target_cuis']
                batch_acc = self.measure_accuracy(final_predictions, target_cuis_batch, mode="Recall@N")

                batch_loss_normalized = batch_loss / update_step
                # accumulated_loss += batch_loss_normalized.item() # 應該在 step 後記錄
                
                if torch.isnan(batch_loss_normalized) or torch.isinf(batch_loss_normalized):
                    print(f"警告: Epoch {ep+1}, Batch {batch_count+1}, 檢測到 NaN 或 Inf 損失，跳過反向傳播。損失值: {batch_loss_normalized.item()}")
                    # 選擇是否需要清零梯度，以防之前的梯度影響下一次有效的 step
                    if (batch_count + 1) % update_step == 0: # 如果剛好在 step 的邊界
                        self.optimizer.zero_grad() # 清零，因為這次 step 不會執行
                    accumulated_loss = 0.0 # 重置，因為這次 step 不會執行
                    batch_count += 1
                    continue # 跳過當前的 backward 和 step

                batch_loss_normalized.backward()
                accumulated_loss += batch_loss_normalized.item() # 在 backward 後記錄
                batch_count += 1

                if batch_count % update_step == 0:
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.gmodel.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    avg_accum_loss = accumulated_loss # 因為 accumulated_loss 是 update_step 個 normalized_loss 的和
                    epoch_loss_train.append(avg_accum_loss)
                    epoch_acc_train.append(batch_acc)
                    train_pbar.set_postfix({'Loss': f'{avg_accum_loss:.4f}', 'Acc': f'{batch_acc:.4f}'})
                    accumulated_loss = 0.0

            avg_epoch_train_loss = np.mean(epoch_loss_train) if epoch_loss_train else float('nan')
            avg_epoch_train_acc = np.mean(epoch_acc_train) if epoch_acc_train else float('nan')
            print(f"\nEpoch {ep+1} Average Training Loss: {avg_epoch_train_loss:.4f}, Average Training Acc: {avg_epoch_train_acc:.4f}")

            # Validation Step
            avg_epoch_dev_loss, avg_epoch_dev_acc = self.validate(dev_data)
            print(f"Epoch {ep+1} Validation Loss: {avg_epoch_dev_loss:.4f}, Validation Acc: {avg_epoch_dev_acc:.4f}")

            if lr_scheduler:
                lr_scheduler.step()
                print(f"LR Scheduler stepped. Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # --- Early Stopping Logic ---
            current_metric_val = None
            if self.early_stopping_metric == 'val_loss':
                current_metric_val = avg_epoch_dev_loss
                # 檢查是否有改善 (損失越低越好)
                if current_metric_val < self.best_metric_val - self.early_stopping_delta:
                    print(f"Validation loss improved ({self.best_metric_val:.4f} --> {current_metric_val:.4f}). Saving model...")
                    self.best_metric_val = current_metric_val
                    self.epochs_no_improve = 0
                    if self.save_model_path:
                        try:
                            torch.save(self.gmodel.state_dict(), self.save_model_path)
                            encoder_save_path = os.path.join(os.path.dirname(self.save_model_path), "encoder.pth")
                            torch.save(self.encoder.state_dict(), encoder_save_path)
                            print(f"Model saved to {self.save_model_path} and {encoder_save_path}")
                        except Exception as e:
                            print(f"Error saving model: {e}")
                else:
                    self.epochs_no_improve += 1
                    print(f"Validation loss did not improve for {self.epochs_no_improve} epoch(s). Best: {self.best_metric_val:.4f}")

            elif self.early_stopping_metric == 'val_acc':
                current_metric_val = avg_epoch_dev_acc
                # 檢查是否有改善 (準確率越高越好)
                if current_metric_val > self.best_metric_val + self.early_stopping_delta:
                    print(f"Validation accuracy improved ({self.best_metric_val:.4f} --> {current_metric_val:.4f}). Saving model...")
                    self.best_metric_val = current_metric_val
                    self.epochs_no_improve = 0
                    if self.save_model_path:
                        try:
                            torch.save(self.gmodel.state_dict(), self.save_model_path)
                            encoder_save_path = os.path.join(os.path.dirname(self.save_model_path), "encoder.pth")
                            torch.save(self.encoder.state_dict(), encoder_save_path)
                            print(f"Model saved to {self.save_model_path} and {encoder_save_path}")
                        except Exception as e:
                            print(f"Error saving model: {e}")
                else:
                    self.epochs_no_improve += 1
                    print(f"Validation accuracy did not improve for {self.epochs_no_improve} epoch(s). Best: {self.best_metric_val:.4f}")

            if self.epochs_no_improve >= self.early_stopping_patience:
                self.early_stop = True
                print(f"\nEarly stopping triggered after {ep+1} epochs due to no improvement for {self.early_stopping_patience} consecutive epochs.")
                break # 跳出訓練循環
            print("-" * 50)

        if not self.early_stop:
            print("Training finished after all epochs.")


    def validate(self, dev_data):
        """Validation loop."""
        print("Running validation...")
        self.mode = 'eval'
        self.encoder.eval()
        self.gmodel.eval()
        epoch_loss_dev = []
        epoch_acc_dev = []
        dev_pbar = tqdm(dev_data, desc="Validation")

        with torch.no_grad():
            for batch in dev_pbar:
                if batch is None: continue

                batch_loss, final_predictions = self.forward_per_batch(batch)

                # Ensure target_cuis_batch uses the hop2 targets for final accuracy calculation
                target_cuis_batch = batch['hop2_target_cuis']
                batch_acc = self.measure_accuracy(final_predictions, target_cuis_batch, mode="Recall@N")

                epoch_loss_dev.append(batch_loss.item())
                epoch_acc_dev.append(batch_acc)
                dev_pbar.set_postfix({
                    'Loss': f'{batch_loss.item():.4f}',
                    'Acc': f'{batch_acc:.4f}'
                    })

        avg_loss = np.mean(epoch_loss_dev) if epoch_loss_dev else 0.0
        avg_acc = np.mean(epoch_acc_dev) if epoch_acc_dev else 0.0
        return avg_loss, avg_acc



# ====================== Main Block ======================
if __name__ =='__main__':

    # --- 常規設置與資源加載 (假設這些已存在或從 args 讀取) ---
    # !!! 確保這些變量已定義並加載好 !!!
    # args = parser.parse_args() # 如果使用 argparse
    TEST_TOKENIZER_PATH = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    GRAPH_FILE = "./drknows/SNOMED_CUI_MAJID_Graph_wSelf.pkl"
    PRETRAIN_VOCAB_EMBEDDING = "./drknows/GraphModel_SNOMED_CUI_Embedding.pkl"
    # CUI_VOCAB_FILE = "./drknows/sm_t047_cui_aui_eng.pkl"
    TRAIN_ANNOTATION_FILE = "./MediQ/mediq_train_annotations.json"
    DEV_ANNOTATION_FILE = './MediQ/mediq_dev_annotations.json'
    tokenizer = AutoTokenizer.from_pretrained(TEST_TOKENIZER_PATH) # 使用測試代碼中的路徑或 args
    encoder = AutoModel.from_pretrained(TEST_TOKENIZER_PATH).to(torch.device('cuda')) # 加載 SapBERT 並移到 GPU
    g = pickle.load(open(GRAPH_FILE, "rb")) # 加載圖譜
    cui_embedding = CuiEmbedding(PRETRAIN_VOCAB_EMBEDDING) # 加載 CUI 嵌入
    #all_edge_mappings = pickle.load(open(ALL_EDGE_MAPPINGS_FILE,"rb")) # 加載邊緣映射
    # cui_vocab = pickle.load(open(CUI_VOCAB_FILE,"rb")) # 加載 CUI 詞彙表
    # cui_weights = json.load(open(CUI_WEIGHT_PATH,"r")) # 如果有 CUI 權重文件
    cui_weights = None # 暫時設為 None

    # --- 動態創建 cui_vocab ---
    print("從圖譜節點動態創建 CUI -> Index 映射...")
    try:
        # 獲取圖譜中所有節點 (CUIs)，並排序以保證映射一致性
        all_graph_cuis = sorted(list(g.nodes()))
        # 創建 CUI 到 index 的映射
        cui_vocab = {cui: i for i, cui in enumerate(all_graph_cuis)}
        # print(f"成功創建 cui_vocab，包含 {len(cui_vocab)} 個 CUIs。")
        # 可選：保存這個動態創建的詞彙表供後續使用
        # with open("generated_cui_vocab_idx.pkl", "wb") as f_out:
        #     pickle.dump(cui_vocab_dynamic, f_out)
    except Exception as e:
        print(f"從圖譜創建 cui_vocab 時出錯: {e}")
        exit()


    # 其他超參數
    hdim = 768
    nums_of_head = 3 # 可能不用於 TriAttn
    top_n = 8 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 300 
    LR = 1e-5 
    intermediate = True # 測試包含中間損失
    contrastive_learning = True # 測試包含對比損失
    batch_size = 16 
    save_model_dir = "./saved_models_mediq"
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    save_model_path = os.path.join(save_model_dir, "gmodel_mediq_best.pth")



    
    print("Instantiating Trainer...")
    trainer = Trainer(
        tokenizer=tokenizer,
        encoder=encoder,
        g=g,
        cui_embedding=cui_embedding,
        hdim=hdim,
        nums_of_head=nums_of_head,
        cui_vocab=cui_vocab,
        # nums_of_hops=2, # 已在內部設為 2
        top_n=top_n,
        device=device,
        nums_of_epochs=epochs, 
        LR=LR,
        cui_weights=cui_weights,
        contrastive_learning=contrastive_learning,
        intermediate=intermediate,
        save_model_path=save_model_path, 
        early_stopping_patience=5, # 例如，如果連續5個epoch沒有改善則停止
        early_stopping_metric='val_loss', # 可以是 'val_loss' 或 'val_acc'
        early_stopping_delta=0.001 # 只有當改善大於此值才算
        # 其他可選參數...
    )
    # trainer.to(device) # Trainer __init__ 內部應已處理
    print("Trainer instantiated successfully.")

    print("\nCreating optimizer...")
    trainer.create_optimizers()
    lr_scheduler= torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=3, gamma=0.6)
    # --- 加載測試數據 ---
    print("\nLoading data batch...")
    try:
        train_dataset = MediQAnnotatedDataset(TRAIN_ANNOTATION_FILE, tokenizer)
        dev_dataset = MediQAnnotatedDataset(DEV_ANNOTATION_FILE, tokenizer)
    except Exception as e:
        print(f"加載數據集時出錯: {e}")
        exit()   
    if len(train_dataset) == 0 or len(dev_dataset) == 0:
        print("Error: dataset is empty!")
        exit()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_mediq_paths)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_mediq_paths)

    
    print("\n" + "="*30)
    print(" STARTING FORMAL TRAINING RUN ")
    print("="*30 + "\n")
    
    try:
        trainer.train(train_loader, dev_loader, lr_scheduler)
    except Exception as e:
        print(f"Error during Training: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*30)
    print(" TRAINING FINISHED ")
    print("="*30 + "\n")




    

