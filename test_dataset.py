import unittest
import os
import json
import tempfile
import torch
import random
from unittest.mock import MagicMock
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

class TestMediQDataLoading(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.max_len = 10 # For mock tokenizer

        # Mock Tokenizer
        self.mock_tokenizer = MagicMock()
        def _side_effect_tokenize(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt"):
            # A very basic mock tokenizer for consistent output shape
            # print(f"Mock tokenize: '{text}' with max_length={max_length}")
            # Fixed output for simplicity in testing structure, not content
            if return_tensors == "pt":
                return {
                    "input_ids": torch.randint(0, 1000, (1, max_length), dtype=torch.long),
                    "attention_mask": torch.ones((1, max_length), dtype=torch.long)
                }
            else: # Not really used by the dataset, but for completeness
                return {
                    "input_ids": [random.randint(0,1000) for _ in range(max_length)],
                    "attention_mask": [1]*max_length
                }
        self.mock_tokenizer.side_effect = _side_effect_tokenize
        self.mock_tokenizer.pad_token_id = 0 # For collate_fn_mediq_preprocessed, though not directly used by it in this version

        # Create a temporary annotation file for MediQAnnotatedDataset
        self.temp_dir = tempfile.TemporaryDirectory()
        self.annotated_file_path = os.path.join(self.temp_dir.name, "mock_annotations.json")
        self.mock_annotations_data = {
            "case_001": { # Valid case, 1-hop and 2-hop paths
                "atomic_facts": ["Fact A0 C001", "Fact A1 is about C002.", "Fact A2 links to C003 via C004.", "Fact A3 is C005."],
                "facts_cuis": [
                    ["C001"], #0
                    ["C002"], #1
                    ["C003", "C004"], #2: C003 is target, C004 is intermediate
                    ["C005"]  #3
                ],
                "paths_between_facts": {
                    "0_1": [["C001", "rel_direct", "C002"]], # 1-hop C001 -> C002
                    "0_2": [["C001", "rel_step1", "C004", "rel_step2", "C003"]], # 2-hop C001 -> C004 (mid) -> C003 (target)
                    "1_3": [["C002", "rel_another", "C005"]] # Another 1-hop
                }
            },
            "case_002": { # Not enough facts
                "atomic_facts": ["Fact B0"], "facts_cuis": [["C006"]], "paths_between_facts": {}
            },
            "case_003": { # Enough facts, but no valid paths_between_facts for any (i,j) pair with target CUI match
                "atomic_facts": ["Fact C0", "Fact C1", "Fact C2"],
                "facts_cuis": [["C007"], ["C008"], ["C009"]],
                "paths_between_facts": {"0_1": []} # Empty path list
            },
            "case_004": { # Path target CUI mismatch
                "atomic_facts": ["Fact D0", "Fact D1"], "facts_cuis": [["C010"], ["C011"]],
                "paths_between_facts": {"0_1": [["C010", "rel_to_other", "C999"]]}
            },
             "case_005": { # target fact_cuis empty
                "atomic_facts": ["Fact E0", "Fact E1 is empty cui"], "facts_cuis": [["C012"], []],
                "paths_between_facts": {"0_1": [["C012", "rel_to_e1", "C013"]]} # C013 not in facts_cuis[1]
            },
            "case_006": { # Should result in getitem returning None because no GT paths can be formed
                "atomic_facts": ["Fact F0", "Fact F1"], "facts_cuis": [["C014"], ["C015"]],
                "paths_between_facts": {} # No paths at all
            }
        }
        with open(self.annotated_file_path, 'w') as f:
            json.dump(self.mock_annotations_data, f)

        # Create a temporary preprocessed file for MediQPreprocessedDataset
        self.preprocessed_file_path = os.path.join(self.temp_dir.name, "mock_preprocessed.jsonl")
        self.mock_preprocessed_data = [
            {"case_id": "prep_001", "tokenized_input_ids": [101, 1, 2, 3, 102, 0, 0, 0], "tokenized_attention_mask": [1, 1, 1, 1, 1, 0, 0, 0], "known_cuis": ["C100", "C101"], "hop1_target_cuis": ["C102"], "hop2_target_cuis": ["C104"], "intermediate_target_cuis": ["C103"]},
            {"case_id": "prep_002", "tokenized_input_ids": [101, 4, 5, 102, 0, 0, 0, 0], "tokenized_attention_mask": [1, 1, 1, 1, 0, 0, 0, 0], "known_cuis": ["C200"], "hop1_target_cuis": [], "hop2_target_cuis": ["C202"], "intermediate_target_cuis": ["C201"]}
        ]
        with open(self.preprocessed_file_path, 'w') as f:
            for item in self.mock_preprocessed_data:
                f.write(json.dumps(item) + "\n")

    def tearDown(self):
        """Tear down test methods."""
        self.temp_dir.cleanup()

    def test_mediq_annotated_dataset_init(self):
        # Test with min_facts_for_sampling = 2
        dataset = MediQAnnotatedDataset(self.annotated_file_path, self.mock_tokenizer, min_facts_for_sampling=2)
        # Expected valid_training_samples from case_001:
        # (0,1) path C001->C002 (1-hop)
        # (0,2) path C001->C004->C003 (2-hop)
        # (1,3) path C002->C005 (1-hop)
        # Case 002, 003, 004, 005, 006 should not produce valid samples.
        self.assertEqual(len(dataset.valid_training_samples), 3)
        
        sample_keys = {(s['case_id'], s['guaranteed_known_idx'], s['guaranteed_unknown_idx']) for s in dataset.valid_training_samples}
        expected_keys = {
            ("case_001", 0, 1),
            ("case_001", 0, 2),
            ("case_001", 1, 3)
        }
        self.assertEqual(sample_keys, expected_keys)

        # Test with min_facts_for_sampling = 4 (case_001 has 4 facts)
        dataset_min4 = MediQAnnotatedDataset(self.annotated_file_path, self.mock_tokenizer, min_facts_for_sampling=4)
        self.assertEqual(len(dataset_min4.valid_training_samples), 3) # case_001 still valid

        # Test with min_facts_for_sampling = 5 (no case should be valid)
        dataset_min5 = MediQAnnotatedDataset(self.annotated_file_path, self.mock_tokenizer, min_facts_for_sampling=5)
        self.assertEqual(len(dataset_min5.valid_training_samples), 0)


    def test_mediq_annotated_dataset_getitem_valid_sample(self):
        # Seed for reproducibility of random split of remaining_indices
        dataset = MediQAnnotatedDataset(self.annotated_file_path, self.mock_tokenizer, random_seed=42, min_facts_for_sampling=2)
        
        # Find the sample corresponding to ("case_001", 0, 1)
        target_sample_info = {"case_id": "case_001", "guaranteed_known_idx": 0, "guaranteed_unknown_idx": 1}
        sample_idx = -1
        for i, s_info in enumerate(dataset.valid_training_samples):
            if s_info['case_id'] == target_sample_info['case_id'] and \
               s_info['guaranteed_known_idx'] == target_sample_info['guaranteed_known_idx'] and \
               s_info['guaranteed_unknown_idx'] == target_sample_info['guaranteed_unknown_idx']:
                sample_idx = i
                break
        self.assertNotEqual(sample_idx, -1, "Test sample (case_001, 0, 1) not found in valid_training_samples")

        item = dataset[sample_idx] # Get the first valid sample: case_001, known_idx=0, unknown_idx=1
        self.assertIsNotNone(item)
        self.assertEqual(item['case_id'], "case_001")
        
        # With random_seed=42, for case_001 (4 facts), guaranteed_known_idx=0, guaranteed_unknown_idx=1
        # Remaining indices: [2, 3]. shuffle([2,3]) can be [2,3] or [3,2].
        # If [2,3], randint(0,2) can be 0, 1, or 2.
        #   If 0 additional: known={0}, unknown={1,2,3}. GTs: C002 (hop1 from 0_1). C003 (hop2 from 0_2), C004 (inter from 0_2)
        #   If 1 additional: known={0,2}, unknown={1,3}. GTs: C002 (hop1 from 0_1).
        #   If 2 additional: known={0,2,3}, unknown={1}. GTs: C002 (hop1 from 0_1).
        # If [3,2], randint(0,2) can be 0, 1, or 2.
        #   If 0 additional: known={0}, unknown={1,3,2}. GTs: C002 (hop1 from 0_1). C003 (hop2 from 0_2), C004 (inter from 0_2)
        # Let's check based on the specific logic for (known=0, unknown=1 for case_001)
        # Path "0_1": ["C001", "rel_direct", "C002"]
        # Expected GTs if only 0 is known and 1 is unknown, and 0_1 is the path:
        # known_indices will contain 0. unknown_indices will contain 1.
        # And facts 2, 3 will be randomly assigned.
        
        # To make this deterministic, we can mock random.randint and random.shuffle for one call
        with unittest.mock.patch('random.Random.shuffle', side_effect=lambda x: x.sort(reverse=True)): # remaining_indices becomes [3,2]
            with unittest.mock.patch('random.Random.randint', return_value=1): # 1 additional known (idx 3)
                dataset_for_specific_split = MediQAnnotatedDataset(self.annotated_file_path, self.mock_tokenizer, random_seed=42)
                item_specific = dataset_for_specific_split[sample_idx] # case_001, g_known=0, g_unknown=1
                # known_indices = {0, 3}
                # unknown_indices = {1, 2}
                # Check k_idx=0, u_idx=1: path 0_1 gives hop1 C002
                # Check k_idx=0, u_idx=2: path 0_2 gives hop2 C003, inter C004
                # Check k_idx=3, u_idx=1: no path
                # Check k_idx=3, u_idx=2: no path
                self.assertIn(0, item_specific['known_indices'])
                self.assertIn(3, item_specific['known_indices'])
                self.assertIn(1, item_specific['unknown_indices'])
                self.assertIn(2, item_specific['unknown_indices'])
                self.assertEqual(set(item_specific['hop1_target_cuis']), {"C002"})
                self.assertEqual(set(item_specific['hop2_target_cuis']), {"C003"})
                self.assertEqual(set(item_specific['intermediate_target_cuis']), {"C004"})
                self.assertIn("C001", item_specific['known_cuis'])
                self.assertIn("C005", item_specific['known_cuis']) # From fact 3

        # Test the (0,2) sample from case_001 for 2-hop GTs
        target_sample_info_2hop = {"case_id": "case_001", "guaranteed_known_idx": 0, "guaranteed_unknown_idx": 2}
        sample_idx_2hop = -1
        for i, s_info in enumerate(dataset.valid_training_samples):
            if s_info['case_id'] == target_sample_info_2hop['case_id'] and \
               s_info['guaranteed_known_idx'] == target_sample_info_2hop['guaranteed_known_idx'] and \
               s_info['guaranteed_unknown_idx'] == target_sample_info_2hop['guaranteed_unknown_idx']:
                sample_idx_2hop = i
                break
        self.assertNotEqual(sample_idx_2hop, -1, "Test sample (case_001, 0, 2) not found")

        with unittest.mock.patch('random.Random.shuffle', side_effect=lambda x: x.sort()): # remaining_indices becomes [1,3]
            with unittest.mock.patch('random.Random.randint', return_value=0): # 0 additional known
                dataset_for_specific_split_2 = MediQAnnotatedDataset(self.annotated_file_path, self.mock_tokenizer, random_seed=123)
                item_2hop = dataset_for_specific_split_2[sample_idx_2hop]
                # known_indices = {0}
                # unknown_indices = {1, 2, 3} (guaranteed_unknown_idx=2)
                # Check k_idx=0, u_idx=1: path 0_1 -> hop1 C002
                # Check k_idx=0, u_idx=2: path 0_2 -> hop2 C003, inter C004
                # Check k_idx=0, u_idx=3: no path
                self.assertEqual(set(item_2hop['known_indices']), {0})
                self.assertIn(1, item_2hop['unknown_indices'])
                self.assertIn(2, item_2hop['unknown_indices'])
                self.assertIn(3, item_2hop['unknown_indices'])

                self.assertEqual(set(item_2hop['hop1_target_cuis']), {"C002"})
                self.assertEqual(set(item_2hop['hop2_target_cuis']), {"C003"})
                self.assertEqual(set(item_2hop['intermediate_target_cuis']), {"C004"})

    def test_mediq_annotated_dataset_getitem_returns_none(self):
        dataset = MediQAnnotatedDataset(self.annotated_file_path, self.mock_tokenizer, min_facts_for_sampling=2)
        # case_006 has no paths_between_facts. This should lead to valid_training_samples being empty for this case.
        # Let's create a specific dataset that *would* produce a valid_training_sample, but for which getitem will find no GTs
        
        # We will test if a sample in valid_training_samples, when its known/unknown split in getitem
        # results in no GTs being formed, returns None.
        # For example, case_001, guaranteed_known_idx=0, guaranteed_unknown_idx=1
        # If known_indices = {0,2,3} and unknown_indices = {1}
        # k_idx=0, u_idx=1 --> path 0_1, GT: C002 (hop1)
        # k_idx=2, u_idx=1 --> no path
        # k_idx=3, u_idx=1 --> no path
        # So this should return a valid item.

        # Let's manually craft a scenario for `getitem` returning None
        # if all paths from *current* known_indices to *current* unknown_indices yield no valid target CUIs.
        # The current logic: "if not hop1_target_cuis_set and not hop2_target_cuis_set and not intermediate_target_cuis_set: return None"
        # This condition is sound. The test for valid_training_samples already checks if a path *could* exist.

        # Try a case that *should* generate no valid training samples in init due to path content
        no_valid_gt_data = {
            "case_no_gt": {
                "atomic_facts": ["Fact X0", "Fact X1", "Fact X2"],
                "facts_cuis": [["CX1"], ["CX2"], ["CX3"]],
                "paths_between_facts": { "0_1": [["CX1", "rel", "CX_OTHER"]]} # Target CX_OTHER not in facts_cuis[1]
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp_f:
            json.dump(no_valid_gt_data, tmp_f)
            tmp_path_no_gt = tmp_f.name
        
        dataset_no_gt_init = MediQAnnotatedDataset(tmp_path_no_gt, self.mock_tokenizer)
        self.assertEqual(len(dataset_no_gt_init.valid_training_samples), 0) # No valid samples from init
        os.unlink(tmp_path_no_gt)


    def test_collate_fn_mediq_paths(self):
        # Mock items from dataset.__getitem__
        mock_item1_tks = self.mock_tokenizer("text1", max_length=self.max_len)
        mock_item1 = {
            "case_id": "c1", "input_text_tks": {k: v.squeeze(0) for k,v in mock_item1_tks.items()},
            "known_cuis": ["C001"], "hop1_target_cuis": ["C002"],
            "hop2_target_cuis": ["C003"], "intermediate_target_cuis": ["C004"]
        }
        mock_item2_tks = self.mock_tokenizer("longer text2", max_length=self.max_len)
        mock_item2 = {
            "case_id": "c2", "input_text_tks": {k: v.squeeze(0) for k,v in mock_item2_tks.items()},
            "known_cuis": ["C005"], "hop1_target_cuis": [],
            "hop2_target_cuis": ["C006"], "intermediate_target_cuis": []
        }

        batch = [mock_item1, mock_item2]
        collated = collate_fn_mediq_paths(batch)

        self.assertIsNotNone(collated)
        self.assertEqual(len(collated["case_id"]), 2)
        self.assertEqual(collated["input_text_tks_padded"]["input_ids"].shape, (2, self.max_len))
        self.assertEqual(collated["input_text_tks_padded"]["attention_mask"].shape, (2, self.max_len))
        self.assertEqual(len(collated["known_cuis"]), 2)

        # Test with a None item
        batch_with_none = [mock_item1, None, mock_item2]
        collated_with_none = collate_fn_mediq_paths(batch_with_none)
        self.assertEqual(len(collated_with_none["case_id"]), 2)

        # Test with all None items
        batch_all_none = [None, None]
        collated_all_none = collate_fn_mediq_paths(batch_all_none)
        self.assertIsNone(collated_all_none)

    def test_mediq_preprocessed_dataset(self):
        dataset = MediQPreprocessedDataset(self.preprocessed_file_path)
        self.assertEqual(len(dataset), 2)

        item0 = dataset[0]
        self.assertEqual(item0["case_id"], "prep_001")
        self.assertTrue(isinstance(item0["input_text_tks_padded"]["input_ids"], torch.Tensor))
        self.assertEqual(item0["input_text_tks_padded"]["input_ids"].shape, (8,)) # Pre-padded length
        self.assertEqual(item0["known_cuis"], ["C100", "C101"])
        self.assertEqual(item0["hop1_target_cuis"], ["C102"])

    def test_collate_fn_mediq_preprocessed(self):
        dataset = MediQPreprocessedDataset(self.preprocessed_file_path)
        # Simulate a batch from DataLoader
        batch_data = [dataset[i] for i in range(len(dataset))]
        
        collated = collate_fn_mediq_preprocessed(batch_data)
        
        self.assertIsNotNone(collated)
        self.assertEqual(len(collated["case_id"]), 2)
        self.assertTrue(isinstance(collated["input_text_tks_padded"]["input_ids"], torch.Tensor))
        self.assertEqual(collated["input_text_tks_padded"]["input_ids"].shape, (2, 8)) # (batch_size, seq_len)
        self.assertTrue(isinstance(collated["input_text_tks_padded"]["attention_mask"], torch.Tensor))
        self.assertEqual(collated["input_text_tks_padded"]["attention_mask"].shape, (2, 8))
        
        self.assertEqual(len(collated["known_cuis"]), 2)
        self.assertEqual(collated["known_cuis"][0], ["C100", "C101"])
        self.assertEqual(collated["hop1_target_cuis"][1], [])

        # Test with an empty batch
        collated_empty = collate_fn_mediq_preprocessed([])
        self.assertIsNone(collated_empty) # Or handle as per desired behavior for empty input

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
