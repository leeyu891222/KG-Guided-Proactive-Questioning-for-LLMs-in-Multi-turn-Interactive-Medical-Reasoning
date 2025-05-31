import unittest
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

# --- MediQAnnotatedDataset (修改後的版本) ---
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


class MockTokenizer:
    def __init__(self):
        self.vocab = {'<pad>': 0, '<unk>': 1, 'fact': 2, 'text': 3}
        self.pad_token_id = 0
    def __call__(self, text, truncation=None, padding=None, max_length=None, return_tensors=None):
        tokens = text.lower().split()
        ids = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        if max_length:
            if padding == "max_length": ids += [self.pad_token_id] * (max_length - len(ids))
        attention_mask = [1] * len(text.lower().split()) + [0] * (max_length - len(text.lower().split()))
        if return_tensors == "pt":
            return {'input_ids': torch.tensor([ids]), 'attention_mask': torch.tensor([attention_mask])}
        return {'input_ids': ids, 'attention_mask': attention_mask}


# 更新後的 Mock 數據 (增加了 case_05)
UPDATED_MOCK_ANNOTATIONS = {
    "case_01": { # 理想情況，混合 hop-1 和 hop-2
        "atomic_facts": ["fact 0", "fact 1", "fact 2"],
        "facts_cuis": [["C001"], ["C002"], ["C003"]],
        "paths_between_facts": {
            "0_1": [["C001", "rel1", "C_INTER_1", "rel2", "C002"]], # 2-hop
            "0_2": [["C001", "rel3", "C003"]]                      # 1-hop
        }
    },
    "case_02": { # 應被過濾
        "atomic_facts": ["single fact"], "facts_cuis": [["C005"]], "paths_between_facts": {}
    },
    "case_03": { # 無效路徑，不應生成樣本
        "atomic_facts": ["fact A", "fact B"], "facts_cuis": [["C006"], ["C007"]],
        "paths_between_facts": {"0_1": [["C006", "rel", "C008_WRONG"]]}
    },
    "case_04": { # 只有 1-hop 路徑
        "atomic_facts": ["fact C", "fact D"], "facts_cuis": [["C009"], ["C010"]],
        "paths_between_facts": {"0_1": [["C009", "rel", "C010"]]}
    },
    "case_05": { # 只有 2-hop 路徑
        "atomic_facts": ["fact E", "fact F"], "facts_cuis": [["C011"], ["C012"]],
        "paths_between_facts": {"0_1": [["C011", "rel", "C_INTER_2", "rel", "C012"]]}
    }
}


class TestUpdatedMediQDataset(unittest.TestCase):

    def setUp(self):
        self.mock_file_path = "updated_mock_annotations.json"
        with open(self.mock_file_path, 'w', encoding='utf-8') as f:
            json.dump(UPDATED_MOCK_ANNOTATIONS, f)
        self.tokenizer = MockTokenizer()

    def tearDown(self):
        if os.path.exists(self.mock_file_path):
            os.remove(self.mock_file_path)

    def test_01_dataset_initialization_v2(self):
        print("\n--- 測試 1 (v2): Dataset 初始化與有效樣本對識別 ---")
        dataset = MediQAnnotatedDataset(self.mock_file_path, self.tokenizer)
        # 預期: case_01(2對), case_04(1對), case_05(1對) -> 共 4 對
        self.assertEqual(len(dataset), 4, "數據集應包含 4 個有效的訓練樣本對")
        print("PASS: Dataset 初始化和樣本對識別正確。")

    def test_02_getitem_gt_separation(self):
        print("\n--- 測試 2 (v2): __getitem__ 的 GT 分離邏輯 ---")
        dataset = MediQAnnotatedDataset(self.mock_file_path, self.tokenizer, random_seed=42)
        
        # 測試 case_04 (只有 1-hop)
        idx_case04 = next(i for i, s in enumerate(dataset.valid_training_samples) if s['case_id'] == 'case_04')
        item_case04 = dataset[idx_case04]
        self.assertIsNotNone(item_case04, "case_04 應返回有效項目")
        self.assertEqual(item_case04['hop1_target_cuis'], ['C010'])
        self.assertEqual(item_case04['hop2_target_cuis'], [])
        self.assertEqual(item_case04['intermediate_target_cuis'], [])
        print("PASS: 只有 1-hop 路徑的樣本 GT 生成正確。")
        
        # 測試 case_05 (只有 2-hop)
        idx_case05 = next(i for i, s in enumerate(dataset.valid_training_samples) if s['case_id'] == 'case_05')
        item_case05 = dataset[idx_case05]
        self.assertIsNotNone(item_case05, "case_05 應返回有效項目")
        self.assertEqual(item_case05['hop1_target_cuis'], [])
        self.assertEqual(item_case05['hop2_target_cuis'], ['C012'])
        self.assertEqual(item_case05['intermediate_target_cuis'], ['C_INTER_2'])
        print("PASS: 只有 2-hop 路徑的樣本 GT 生成正確。")

        # 測試 case_01 (混合)
        idx_case01_p1 = next(i for i, s in enumerate(dataset.valid_training_samples) if s['case_id'] == 'case_01' and s['guaranteed_known_idx'] == 0 and s['guaranteed_unknown_idx'] == 1)
        # 假設 random_seed=42 使得 known={0}, unknown={1,2}
        item_case01_p1 = dataset[idx_case01_p1] 
        self.assertIsNotNone(item_case01_p1)
        # known={0}, unknown={1,2} -> GT 來自 0->1 和 0->2
        # 0->1: hop2 GT=C002, intermediate GT=C_INTER_1
        # 0->2: hop1 GT=C003
        self.assertCountEqual(item_case01_p1['hop1_target_cuis'], ['C003'])
        self.assertCountEqual(item_case01_p1['hop2_target_cuis'], ['C002'])
        self.assertCountEqual(item_case01_p1['intermediate_target_cuis'], ['C_INTER_1'])
        print("PASS: 混合路徑的樣本 GT 生成正確。")

    def test_03_collate_fn_with_new_key(self):
        print("\n--- 測試 3 (v2): collate_fn 對新鍵的處理 ---")
        dataset = MediQAnnotatedDataset(self.mock_file_path, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn_mediq_paths)
        
        batch = next(iter(dataloader))
        
        self.assertIn('intermediate_target_cuis', batch, "Collated batch 應包含 'intermediate_target_cuis' 鍵")
        self.assertIsInstance(batch['intermediate_target_cuis'], list, "'intermediate_target_cuis' 應為 list")
        self.assertEqual(len(batch['intermediate_target_cuis']), 4, "批次中應有 4 個樣本的 intermediate_target_cuis 列表")
        print("PASS: collate_fn 正確處理了新增的 'intermediate_target_cuis' 鍵。")

if __name__ == '__main__':
    # 運行測試
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestUpdatedMediQDataset))
    runner = unittest.TextTestRunner()
    runner.run(suite)