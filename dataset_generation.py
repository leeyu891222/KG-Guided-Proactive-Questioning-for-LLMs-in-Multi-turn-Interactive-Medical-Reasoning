import json
import random
import os
from tqdm import tqdm
from transformers import AutoTokenizer

# --- 配置參數 ---
# 輸入的原始標註文件
INPUT_ANNOTATION_FILE = "./MediQ/mediq_test_annotations_bm25_20.json" 
# 輸出的預處理後文件【基礎名稱】，不再包含擴展名
OUTPUT_PREPROCESSED_FILE_BASE = "./Evaluate/mediq_test_preprocessed"
# Tokenizer 路徑
TOKENIZER_PATH = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
# 一個案例至少需要多少個事實才能用於生成樣本
MIN_FACTS_FOR_SAMPLING = 2  
# 對於每個有效的 (guaranteed_known, guaranteed_unknown) 對，生成多少個不同的已知/未知集劃分樣本
SAMPLES_PER_GUARANTEED_PAIR = 1 
# 隨機種子，確保每次預處理結果一致
RANDOM_SEED = 2023       

# --- 初始化 Tokenizer ---
print(f"正在載入 Tokenizer 從: {TOKENIZER_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)
print("Tokenizer 載入完成。")

# --- 初始化隨機數生成器 ---
random_gen = random.Random(RANDOM_SEED)

def preprocess_data(input_file, output_file, tokenizer_instance, 
                    min_facts=MIN_FACTS_FOR_SAMPLING, 
                    samples_per_pair=SAMPLES_PER_GUARANTEED_PAIR):
    print(f"開始預處理文件: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            all_annotations = json.load(f)
    except FileNotFoundError:
        print(f"錯誤：找不到標註文件 {input_file}。")
        return
    except Exception as e:
        print(f"載入標註文件時發生錯誤: {e}")
        return

    num_original_cases = len(all_annotations)
    num_processed_cases = 0
    total_generated_samples = 0
    
    # 打開輸出文件準備寫入
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for case_id, data in tqdm(all_annotations.items(), desc="正在處理案例"):
            atomic_facts = data.get("atomic_facts", [])
            facts_cuis = data.get("facts_cuis", [])
            paths_between_facts = data.get("paths_between_facts", {})
            num_facts = len(atomic_facts)

            if num_facts < min_facts or len(facts_cuis) != num_facts:
                continue
            
            num_processed_cases += 1
            
            valid_guaranteed_pairs = []
            for i in range(num_facts):
                for j in range(num_facts):
                    if i == j: continue
                    path_key = f"{i}_{j}"
                    if path_key in paths_between_facts and paths_between_facts[path_key]:
                        if facts_cuis[j]:
                            has_valid_target = False
                            for path_data in paths_between_facts[path_key]:
                                if not path_data or not isinstance(path_data, list): continue
                                target_cui_in_path = path_data[-1]
                                if isinstance(target_cui_in_path, str) and target_cui_in_path.startswith('C') and \
                                   target_cui_in_path in facts_cuis[j]:
                                    has_valid_target = True
                                    break
                            if has_valid_target:
                                valid_guaranteed_pairs.append((i, j))
            
            if not valid_guaranteed_pairs:
                continue

            for gk_idx, gu_idx in valid_guaranteed_pairs:
                for _ in range(samples_per_pair):
                    known_indices = {gk_idx}
                    unknown_indices = {gu_idx}
                    remaining_indices = list(set(range(num_facts)) - known_indices - unknown_indices)
                    random_gen.shuffle(remaining_indices)

                    if remaining_indices:
                        num_additional_known = random_gen.randint(0, len(remaining_indices))
                        known_indices.update(remaining_indices[:num_additional_known])
                        unknown_indices.update(remaining_indices[num_additional_known:])
                    
                    sorted_known_indices = sorted(list(known_indices))
                    sorted_unknown_indices = sorted(list(unknown_indices))

                    known_texts_list = [atomic_facts[i] for i in sorted_known_indices]
                    known_texts_combined = " ".join(known_texts_list) if known_texts_list else "N/A"
                    
                    tokenized_input = tokenizer_instance(known_texts_combined,
                                                         truncation=True, 
                                                         padding='max_length',
                                                         max_length=512,
                                                         return_tensors=None)
                    
                    known_cuis_list_flat = []
                    for k_i in sorted_known_indices:
                        current_fact_cuis = facts_cuis[k_i]
                        if isinstance(current_fact_cuis, list):
                            known_cuis_list_flat.extend(current_fact_cuis)
                    known_cuis_list_unique = list(set(known_cuis_list_flat))
                    
                    # --- 【修改點 1】確保 target 不是 known CUI ---
                    known_cuis_set = set(known_cuis_list_unique)

                    hop1_target_cuis_set = set()
                    hop2_target_cuis_set = set()
                    intermediate_target_cuis_set = set()

                    for k_fact_idx in sorted_known_indices:
                        for u_fact_idx in sorted_unknown_indices:
                            path_key_dynamic = f"{k_fact_idx}_{u_fact_idx}"
                            if path_key_dynamic in paths_between_facts:
                                for path_data in paths_between_facts[path_key_dynamic]:
                                    if not path_data or not isinstance(path_data, list): continue
                                    path_len = len(path_data)
                                    target_cui_in_path = path_data[-1]
                                    is_valid_cui = isinstance(target_cui_in_path, str) and target_cui_in_path.startswith('C')
                                    
                                    if is_valid_cui and isinstance(facts_cuis[u_fact_idx], list) and \
                                       target_cui_in_path in facts_cuis[u_fact_idx]:

                                        # 在這裡加入過濾邏輯
                                        if target_cui_in_path in known_cuis_set:
                                            continue # 如果最終目標是已知CUI，則跳過此路徑

                                        if path_len == 3:
                                            hop1_target_cuis_set.add(target_cui_in_path)
                                        elif path_len == 5:
                                            intermediate_cui = path_data[2]
                                            # 中間節點可以為已知，所以不過濾
                                            intermediate_target_cuis_set.add(intermediate_cui)
                                            hop2_target_cuis_set.add(target_cui_in_path)
                    
                    if not hop1_target_cuis_set and not hop2_target_cuis_set:
                        continue

                    preprocessed_sample = {
                        "case_id": case_id,
                        "known_indices": sorted_known_indices,
                        "unknown_indices": sorted_unknown_indices,
                        "tokenized_input_ids": tokenized_input["input_ids"],
                        "tokenized_attention_mask": tokenized_input["attention_mask"],
                        "known_cuis": known_cuis_list_unique,
                        "hop1_target_cuis": list(hop1_target_cuis_set),
                        "hop2_target_cuis": list(hop2_target_cuis_set),
                        "intermediate_target_cuis": list(intermediate_target_cuis_set)
                    }
                    
                    outfile.write(json.dumps(preprocessed_sample) + "\n")
                    total_generated_samples += 1
            
            if (num_processed_cases + 1) % 100 == 0:
                tqdm.write(f"已處理 {num_processed_cases + 1}/{num_original_cases} 個案例，生成 {total_generated_samples} 個訓練樣本。")

    print(f"\n預處理完成！")
    print(f"原始案例數: {num_original_cases}")
    print(f"實際處理的案例數 (符合最小事實數要求): {num_processed_cases}")
    print(f"總共生成的訓練樣本數: {total_generated_samples}")
    print(f"預處理結果已保存到: {output_file}")


# --- 主執行函數 ---
if __name__ == "__main__":
    
    # --- 【修改點 2】動態生成包含 random seed 的輸出文件名 ---
    base, ext = os.path.splitext(OUTPUT_PREPROCESSED_FILE_BASE)
    # 如果基礎名本身就是 .jsonl，os.splitext 可能會誤判，我們確保基礎名乾淨
    if base.endswith('.jsonl'):
        base = base[:-6]
    
    # 構造新的文件名，例如：./MediQ/mediq_dev_preprocessed_seed2023.jsonl
    final_output_filename = f"{base}_seed{RANDOM_SEED}.jsonl"
    # --- 修改結束 ---

    preprocess_data(INPUT_ANNOTATION_FILE, final_output_filename, tokenizer)