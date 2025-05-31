import cProfile
import pstats
import random
import json
from torch.utils.data import Dataset # 確保 Dataset 被導入
from line_profiler import LineProfiler # 用於 line_profiler

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
    @profile
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
    def __init__(self, max_length=512, pad_token_id=0):
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        # 模擬一些詞彙
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "fact": 10, "text": 11, "patient": 12, "cui": 13}
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

    def __call__(self, text_input, truncation=True, padding='max_length', max_length=None, return_tensors=None, **kwargs):
        if max_length is None:
            max_length = self.max_length
        
        # 簡單的按空格分詞
        tokens = str(text_input).lower().split()
        
        input_ids = [self.vocab.get(token, self.vocab["[UNK]"]) for token in tokens]

        if truncation and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]

        attention_mask = [1] * len(input_ids)

        if padding == 'max_length' and len(input_ids) < max_length:
            pad_length = max_length - len(input_ids)
            input_ids.extend([self.pad_token_id] * pad_length)
            attention_mask.extend([0] * pad_length)
        
        if return_tensors == 'pt':
            # Dataset 的 __getitem__ 通常不需要批次維度，squeeze(0) 是在 collate_fn 之後，
            # 但 tokenizer 可能直接返回帶批次維度的，所以這裡保持與 Hugging Face 一致
            return {
                "input_ids": torch.tensor([input_ids]),
                "attention_mask": torch.tensor([attention_mask])
            }
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    
    
    
    
# --- 分析配置 ---
ANNOTATION_FILE_PATH = "./MediQ/mediq_train_annotations_bm25_20.json"  # !!! 修改為您的標註文件路徑 !!!
NUM_GETITEM_CALLS = 100  # 對 __getitem__ 的總調用次數
NUM_SAMPLES_TO_TEST = 5   # 從數據集中隨機選擇多少個不同的樣本進行測試

# --- 準備 Dataset 實例 ---
print("正在初始化 Dataset 和 Tokenizer...")
# tokenizer = MockTokenizer() # 使用 Mock Tokenizer
# 或者使用您的真實 Tokenizer:
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext", use_fast=True)


# !!! 確保您已將 MediQAnnotatedDataset 的定義粘貼到上方 !!!
# 如果 MediQAnnotatedDataset 未定義，以下代碼會出錯
if 'MediQAnnotatedDataset' not in globals():
    print("錯誤: MediQAnnotatedDataset 類未定義。請將其定義粘貼到腳本中。")
    exit()

try:
    dataset = MediQAnnotatedDataset(ANNOTATION_FILE_PATH, tokenizer)
except Exception as e:
    print(f"初始化 Dataset 時出錯: {e}")
    exit()

if len(dataset) == 0:
    print("錯誤: Dataset 初始化後為空，無法進行性能分析。請檢查標註文件和 Dataset 初始化邏輯。")
    exit()

# 隨機選取一些樣本索引進行測試
if len(dataset) < NUM_SAMPLES_TO_TEST:
    print(f"警告: Dataset 中的樣本數 ({len(dataset)}) 少於請求的測試樣本數 ({NUM_SAMPLES_TO_TEST})。將使用所有可用樣本。")
    sample_indices_to_test = list(range(len(dataset)))
else:
    sample_indices_to_test = random.sample(range(len(dataset)), NUM_SAMPLES_TO_TEST)

print(f"將對 {len(sample_indices_to_test)} 個樣本索引進行測試，總共調用 __getitem__ {NUM_GETITEM_CALLS * len(sample_indices_to_test)} 次。")
print(f"測試的樣本索引: {sample_indices_to_test}")


# --- 使用 cProfile 進行分析 ---
def profile_with_cprofile():
    print("\n" + "="*30)
    print("開始使用 cProfile 進行性能分析...")
    print("="*30)
    
    profiler = cProfile.Profile()
    profiler.enable()

    for i in range(NUM_GETITEM_CALLS):
        for sample_idx in sample_indices_to_test:
            try:
                _ = dataset[sample_idx] # 調用 __getitem__
            except Exception as e:
                print(f"調用 dataset[{sample_idx}] 時出錯: {e}")
                break
        if i % (NUM_GETITEM_CALLS // 10 if NUM_GETITEM_CALLS >=10 else 1) == 0 :
             print(f"cProfile: 已完成 {i+1}/{NUM_GETITEM_CALLS} 輪調用...")


    profiler.disable()
    
    print("\n--- cProfile 分析結果 (按累積耗時排序) ---")
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(30) # 打印耗時最高的30個函數

    print("\n--- cProfile 分析結果 (按內部耗時排序) ---")
    stats_tottime = pstats.Stats(profiler).sort_stats('tottime')
    stats_tottime.print_stats(30)
    print("="*30)
    print("cProfile 分析完成。")
    print("="*30)

# --- 使用 line_profiler 進行分析的說明 ---
def explain_line_profiler():
    print("\n" + "="*30)
    print("關於 line_profiler 的使用說明")
    print("="*30)
    print("1. 確認您已經安裝了 line_profiler: pip install line_profiler")
    print("2. 在您的 MediQAnnotatedDataset 類的 __getitem__ 方法定義上方，添加裝飾器 @profile。")
    print("   示例:")
    print("   class MediQAnnotatedDataset(Dataset):")
    print("       # ... (其他方法) ...")
    print("       from line_profiler import profile # 可以放在方法內部或類外部，但kernprof會處理")
    print("       @profile")
    print("       def __getitem__(self, index):")
    print("           # ... 您的 __getitem__ 實現 ...")
    print("3. 保存修改後的包含 @profile 裝飾器的 Python 文件 (例如，就保存為本文件名 profile_getitem.py)。")
    print("4. 在命令行中，運行以下命令 (假設您的腳本名為 profile_getitem.py):")
    print("   kernprof -l -v profile_getitem.py")
    print("5. kernprof 會運行您的腳本（包括下面的 `run_getitem_for_line_profiler` 函數），")
    print("   然後打印出 __getitem__ 方法中每一行的詳細耗時信息。")
    print("   重點關注 '% Time' 和 'Time per hit' 較高的行。")
    print("="*30)

def run_getitem_for_line_profiler():
    # 這個函數是為了被 kernprof 調用時實際執行 __getitem__
    # 當您使用 kernprof 時，它會自動執行這個腳本。
    # 如果您只想運行 cProfile，可以註釋掉對此函數的調用。
    print("\n" + "="*30)
    print("正在為 line_profiler 準備執行 __getitem__ 調用...")
    print("如果您看到了這個訊息但沒有使用 'kernprof' 運行，則不會有 line_profiler 的輸出。")
    print("="*30)
    for i in range(NUM_GETITEM_CALLS): # NUM_GETITEM_CALLS 可以設小一點，例如10-50，避免line_profiler太慢
        for sample_idx in sample_indices_to_test:
            try:
                _ = dataset[sample_idx]
            except Exception as e:
                print(f"調用 dataset[{sample_idx}] 時 (for line_profiler) 出錯: {e}")
                # 為了不中斷 line_profiler 的其他分析，可以選擇 continue 或 break
                break 
        if i % (NUM_GETITEM_CALLS // 10 if NUM_GETITEM_CALLS >=10 else 1) == 0 :
             print(f"line_profiler run: 已完成 {i+1}/{NUM_GETITEM_CALLS} 輪調用...")
    print("為 line_profiler 執行的 __getitem__ 調用完成。")


if __name__ == "__main__":
    # --- 執行 cProfile ---
    profile_with_cprofile()

    # --- 執行 line_profiler (的準備函數) ---
    # 當您使用 `kernprof -l -v profile_getitem.py` 運行時，
    # kernprof 會確保 @profile 裝飾的方法被追蹤，並執行整個腳本。
    # 這個 run_getitem_for_line_profiler() 函數的調用是為了確保 __getitem__ 被足夠次數地執行。
    explain_line_profiler()
    run_getitem_for_line_profiler() # 您可以在只想運行cProfile時註釋掉這行