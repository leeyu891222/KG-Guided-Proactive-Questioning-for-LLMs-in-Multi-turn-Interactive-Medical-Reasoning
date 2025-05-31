import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm # 用於顯示加載進度
from torch.utils.data import Dataset, DataLoader

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

# --- 示例使用 ---
if __name__ == '__main__':
    # 假設您已經運行了之前的預處理腳本，並生成了 OUTPUT_PREPROCESSED_FILE
    PREPROCESSED_FILE = "./MediQ/mediq_train_preprocessed.jsonl" # 使用您的實際路徑

    # 創建一個 Mock Tokenizer 實例，主要用於獲取可能的 padding 信息 (如果需要)
    # 如果您的數據在預處理時已經 padding 到了固定長度，這裡的 tokenizer 可能不是嚴格必需的
    # 但 DataLoader 的 collate_fn 可能會用到它，或者 __getitem__ 需要它來轉換為張量
    # 在這個版本中，__getitem__ 自己轉換為張量，collate_fn 只是堆疊
    
    # from transformers import AutoTokenizer
    # tokenizer_instance = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    
    print("正在創建 Dataset 實例...")
    try:
        # 注意：由於 __getitem__ 自己處理張量轉換，這裡的 tokenizer_for_padding_info 變得不那麼重要
        # 除非您希望 collate_fn 更通用。
        preprocessed_dataset = MediQPreprocessedDataset(PREPROCESSED_FILE) 
    except Exception as e:
        print(f"創建 Dataset 實例時出錯: {e}")
        exit()

    if len(preprocessed_dataset) > 0:
        print(f"\n成功從 Dataset 獲取第一個樣本:")
        first_sample = preprocessed_dataset[0]
        for key, value in first_sample.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        print(f"    {sub_key}: shape={sub_value.shape}, dtype={sub_value.dtype}")
                    else:
                        print(f"    {sub_key}: {sub_value}")
            elif isinstance(value, torch.Tensor):
                 print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {value}")

        print(f"\n使用 DataLoader 進行測試 (batch_size=2)...")
        # 使用新的 collate_fn
        dataloader = DataLoader(preprocessed_dataset, batch_size=2, collate_fn=collate_fn_mediq_preprocessed)
        
        try:
            for i, batch_data in enumerate(dataloader):
                if i >= 2: # 只打印前兩個批次
                    break
                if batch_data is None:
                    print("DataLoader 返回了一個空批次。")
                    continue

                print(f"\n--- Batch {i+1} ---")
                for key, value in batch_data.items():
                    if key == "input_text_tks_padded" and isinstance(value, dict):
                        print(f"  {key}:")
                        for sub_key, sub_value in value.items():
                             if isinstance(sub_value, torch.Tensor):
                                print(f"    {sub_key}: shape={sub_value.shape}, dtype={sub_value.dtype}, device={sub_value.device}")
                             else:
                                print(f"    {sub_key}: {sub_value}")
                    elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor): # 處理張量列表 (儘管此collate不應產生)
                        print(f"  {key}: list of {len(value)} tensors, first shape={value[0].shape}")
                    elif isinstance(value, torch.Tensor):
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
                    else: # 通常是列表的列表 (例如 known_cuis) 或列表 (例如 case_id)
                        print(f"  {key}: (length {len(value) if isinstance(value,list) else 'N/A'}) {value}")
        except Exception as e:
            print(f"處理 DataLoader 批次時出錯: {e}")
            import traceback
            traceback.print_exc()

    else:
        print("Dataset 為空，無法執行示例使用。")