import json
import pickle
import math
from tqdm import tqdm
import numpy as np
from collections import Counter, defaultdict
import os

# --- Configuration ---
ANNOTATION_FILE_PATH = "./MediQ/mediq_test_annotations_semantic_k2.json"  # Replace
CUI_TO_TEXT_PKL_PATH = "./drknows/sm_t047_cui_aui_eng.pkl"   # Replace
FILTERED_ANNOTATION_OUTPUT_PATH = "./MediQ/mediq_test_annotations_bm25_20.json"
CUI_WEIGHTS_OUTPUT_PATH = "./cui_bm25_weights_test.json"

RETENTION_RATIO_TWO_HOP = 0.2 # Keep top 20% of 2-hop paths per case
BM25_K1 = 1.5  # Typical BM25 parameter k1
BM25_B = 0.75   # Typical BM25 parameter b

# --- Helper Functions ---

def get_cui_text_representation(cui, cui_to_text_map):
    """Gets the first preferred text for a CUI."""
    if cui in cui_to_text_map and cui_to_text_map[cui]:
        return cui_to_text_map[cui][0][1]
    return cui

# --- BM25 Calculation Functions ---

def calculate_term_frequency_in_case(cui_target, case_facts_cuis_list):
    """
    Calculates TF: how many unique facts in this case contain the cui_target.
    """
    count = 0
    for fact_cuis in case_facts_cuis_list:
        if cui_target in fact_cuis:
            count += 1
    return count

def calculate_bm25_idf(cui_target_df, total_num_cases):
    """
    Calculates IDF component for BM25.
    Using a common formula: log( (N - n + 0.5) / (n + 0.5) + 1 )
    """
    numerator = total_num_cases - cui_target_df + 0.5
    denominator = cui_target_df + 0.5
    # Adding 1 inside the log is a common variation, or adding small epsilon if N-n+0.5 is 0
    # Let's use the variant that avoids issues with n=N or n=0 slightly differently
    # A simpler variant: log(1 + (N - n + 0.5) / (n + 0.5))
    # Or Okapi BM25 IDF: log((N - n + 0.5) / (n + 0.5)) - ensure n > 0
    if denominator == 0: # Should not happen if cui_target_df is from observed CUIs.
        # For a CUI with DF=0 (not in training set), its IDF would be max.
        # log( (N+0.5) / 0.5 )
        return math.log((total_num_cases + 0.5) / 0.5) # High IDF for unseen

    return math.log(1 + (total_num_cases - cui_target_df + 0.5) / (cui_target_df + 0.5))


def calculate_bm25_score(tf, idf, doc_len, avg_doc_len, k1, b):
    """
    Calculates the BM25 score for a term in a document.
    """
    numerator = tf * (k1 + 1)
    denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
    if denominator == 0: # Avoid division by zero if tf is 0 and k1*(...) is 0
        return 0.0 
    return idf * (numerator / denominator)

# --- Main Script ---
def main():
    print("🚀 Starting BM25-based GT Path Filtering Process...")

    # 1. Load Data
    print("\n--- 📂 Loading Data ---")
    try:
        with open(ANNOTATION_FILE_PATH, 'r', encoding='utf-8') as f:
            annotations_data = json.load(f)
        print(f"✅ Loaded annotations from: {ANNOTATION_FILE_PATH} ({len(annotations_data)} cases)")

        with open(CUI_TO_TEXT_PKL_PATH, 'rb') as f:
            cui_to_text_map = pickle.load(f)
        print(f"✅ Loaded CUI-to-text mappings from: {CUI_TO_TEXT_PKL_PATH} ({len(cui_to_text_map)} CUIs in map)")

    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not find a required file. Missing: {e.filename}")
        return
    except Exception as e:
        print(f"❌ ERROR loading data: {e}")
        return

    # 2. Pre-calculate Statistics for BM25
    print("\n--- 📊 Pre-calculating BM25 Statistics ---")
    master_cui_list = set(cui_to_text_map.keys())
    doc_freqs = Counter() # DF: CUI -> number of cases it appears in
    doc_lengths = {}    # dl: case_id -> number of atomic_facts
    all_cuis_in_dataset = set()

    for case_id, case_content in annotations_data.items():
        doc_lengths[case_id] = len(case_content.get("atomic_facts", []))
        
        case_unique_cuis = set()
        for fact_cuis_list in case_content.get("facts_cuis", []):
            for cui in fact_cuis_list:
                case_unique_cuis.add(cui)
                all_cuis_in_dataset.add(cui)
        
        for cui in case_unique_cuis:
            doc_freqs[cui] += 1

    total_num_cases = len(annotations_data)
    if total_num_cases == 0:
        print("❌ ERROR: No cases found in annotation data. Exiting.")
        return
        
    avg_doc_length = sum(doc_lengths.values()) / total_num_cases if total_num_cases > 0 else 0
    
    ood_cuis = master_cui_list - all_cuis_in_dataset
    num_ood_cuis = len(ood_cuis)
    print(f"✅ Statistics calculated: Total Cases (N)={total_num_cases}, Avg Doc Length (avgdl)={avg_doc_length:.2f}")
    print(f"  Total unique CUIs in CUI-map: {len(master_cui_list)}")
    print(f"  Total unique CUIs observed in dataset: {len(all_cuis_in_dataset)}")
    print(f"  Number of OOD CUIs (in map but not in dataset facts_cuis): {num_ood_cuis}")

    # Calculate IDF for all CUIs in the master list
    idf_scores = {}
    for cui in master_cui_list:
        df = doc_freqs.get(cui, 0) # If CUI not in doc_freqs, its DF is 0
        idf_scores[cui] = calculate_bm25_idf(df, total_num_cases)
    print(f"  IDF scores calculated for {len(idf_scores)} master CUIs.")


    # 3. Calculate BM25 scores for each CUI in each case it appears & store for cui_weights
    cui_bm25_scores_per_case = defaultdict(list) # cui -> [bm25_score_in_case1, bm25_score_in_case2, ...]
    
    print("\n--- 💯 Calculating BM25 scores for CUIs within each case ---")
    for case_id, case_content in tqdm(annotations_data.items(), desc="Calculating CUI BM25s"):
        case_doc_len = doc_lengths[case_id]
        current_case_facts_cuis = case_content.get("facts_cuis", [])
        
        # Get all unique CUIs appearing in this specific case
        unique_cuis_in_this_case = set()
        for fact_cuis_list in current_case_facts_cuis:
            for cui in fact_cuis_list:
                unique_cuis_in_this_case.add(cui)
        
        for cui in unique_cuis_in_this_case:
            tf = calculate_term_frequency_in_case(cui, current_case_facts_cuis)
            idf = idf_scores.get(cui, calculate_bm25_idf(0, total_num_cases)) # Use precomputed IDF, or recompute for OOD if needed
            
            bm25 = calculate_bm25_score(tf, idf, case_doc_len, avg_doc_length, BM25_K1, BM25_B)
            cui_bm25_scores_per_case[cui].append(bm25)


    # 4. Filter 2-hop paths based on endpoint CUI's BM25 score within its case
    print("\n--- 🔍 Filtering 2-Hop Paths per Case ---")
    filtered_annotations_data = {}
    stats_before_filter = {"total_1hop": 0, "total_2hop": 0}
    stats_after_filter = {"total_1hop": 0, "total_2hop": 0}
    case_threshold_scores = []

    for case_id, case_content in tqdm(annotations_data.items(), desc="Filtering Paths"):
        filtered_case_content = {
            "atomic_facts": case_content.get("atomic_facts", []),
            "facts_cuis": case_content.get("facts_cuis", []),
            "paths_between_facts": {}
        }
        case_doc_len = doc_lengths[case_id]
        current_case_facts_cuis = case_content.get("facts_cuis", [])
        
        all_2_hop_paths_in_this_case_with_scores = []

        original_paths_for_case = case_content.get("paths_between_facts", {})
        for path_key, path_list in original_paths_for_case.items():
            if path_key not in filtered_case_content["paths_between_facts"]:
                filtered_case_content["paths_between_facts"][path_key] = []

            for path_data in path_list:
                if not isinstance(path_data, list) or not path_data: continue

                path_len_nodes = sum(1 for item in path_data if isinstance(item, str) and item.startswith('C'))
                num_relations = len(path_data) - path_len_nodes
                is_one_hop = (path_len_nodes == 2 and num_relations == 1 and len(path_data) == 3)
                is_two_hop = (path_len_nodes == 3 and num_relations == 2 and len(path_data) == 5)

                if is_one_hop:
                    stats_before_filter["total_1hop"] += 1
                    filtered_case_content["paths_between_facts"][path_key].append(path_data)
                    stats_after_filter["total_1hop"] += 1
                elif is_two_hop:
                    stats_before_filter["total_2hop"] += 1
                    endpoint_cui = path_data[-1]
                    
                    # Calculate BM25 for this endpoint_cui in this specific case
                    tf_endpoint = calculate_term_frequency_in_case(endpoint_cui, current_case_facts_cuis)
                    idf_endpoint = idf_scores.get(endpoint_cui, calculate_bm25_idf(0, total_num_cases)) # Default to high IDF if not in master
                    
                    bm25_score_endpoint = calculate_bm25_score(
                        tf_endpoint, idf_endpoint, case_doc_len, avg_doc_length, BM25_K1, BM25_B
                    )
                    all_2_hop_paths_in_this_case_with_scores.append(
                        {"path_key": path_key, "path_data": path_data, "bm25_score": bm25_score_endpoint}
                    )
        
        # Now, for this case, sort all its 2-hop paths and keep top RETENTION_RATIO_TWO_HOP
        if all_2_hop_paths_in_this_case_with_scores:
            all_2_hop_paths_in_this_case_with_scores.sort(key=lambda x: x["bm25_score"], reverse=True)
            
            num_to_keep = math.ceil(len(all_2_hop_paths_in_this_case_with_scores) * RETENTION_RATIO_TWO_HOP)
            
            if num_to_keep == 0 and len(all_2_hop_paths_in_this_case_with_scores) > 0 and RETENTION_RATIO_TWO_HOP > 0:
                 num_to_keep = 1 # Ensure at least one is kept if ratio > 0 and paths exist

            selected_2_hop_for_case = all_2_hop_paths_in_this_case_with_scores[:num_to_keep]
            
            if selected_2_hop_for_case:
                # Record the score of the last kept path as the threshold for this case
                threshold_score_for_case = selected_2_hop_for_case[-1]["bm25_score"]
                case_threshold_scores.append(threshold_score_for_case)
                
                for item in selected_2_hop_for_case:
                    original_path_key = item["path_key"]
                    if original_path_key not in filtered_case_content["paths_between_facts"]:
                         # This should ideally not happen if 1-hops are added first
                         filtered_case_content["paths_between_facts"][original_path_key] = []
                    filtered_case_content["paths_between_facts"][original_path_key].append(item["path_data"])
                    stats_after_filter["total_2hop"] += 1
            elif all_2_hop_paths_in_this_case_with_scores: # Paths existed, but num_to_keep was 0 (e.g. ratio is 0)
                case_threshold_scores.append(float('inf')) # effectively no paths kept by score
            # If no 2-hop paths to begin with, no threshold recorded

        # Remove path_keys if they ended up with empty path lists after filtering
        keys_to_delete = [pk for pk, pl in filtered_case_content["paths_between_facts"].items() if not pl]
        for pk_del in keys_to_delete:
            del filtered_case_content["paths_between_facts"][pk_del]

        filtered_annotations_data[case_id] = filtered_case_content
        
    # 5. Save Filtered Annotations
    print("\n--- 💾 Saving Filtered Annotations ---")
    try:
        with open(FILTERED_ANNOTATION_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(filtered_annotations_data, f, indent=2)
        print(f"✅ Filtered annotations saved to: {FILTERED_ANNOTATION_OUTPUT_PATH}")
    except Exception as e:
        print(f"❌ ERROR saving filtered annotations: {e}")

    # 6. Generate cui_weight_file (Average BM25 score)
    print("\n--- ⚖️ Generating CUI Weight File (Average BM25) ---")
    cui_average_bm25_scores = {}
    for cui, scores_list in cui_bm25_scores_per_case.items():
        if scores_list:
            cui_average_bm25_scores[cui] = sum(scores_list) / len(scores_list)
        else: # Should not happen if cui is in this dict
            cui_average_bm25_scores[cui] = 0.0 
            
    # For CUIs in master_cui_list but not in any case (DF=0), their avg BM25 is effectively 0
    # based on observation. If you want to assign them a high IDF-based score, that's a different policy.
    # For Option 2 (Average BM25), DF=0 means it was never observed, so average is undefined or 0.
    for cui in master_cui_list:
        if cui not in cui_average_bm25_scores:
            cui_average_bm25_scores[cui] = 0.0 # Assign 0 for CUIs never seen in dataset cases

    try:
        with open(CUI_WEIGHTS_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(cui_average_bm25_scores, f, indent=2)
        print(f"✅ CUI weights (average BM25) saved to: {CUI_WEIGHTS_OUTPUT_PATH}")
    except Exception as e:
        print(f"❌ ERROR saving CUI weights: {e}")


    # 7. Print Final Statistics
    print("\n--- 📊 Final Statistics ---")
    print("Original Path Counts:")
    print(f"  Total One-Hop Paths: {stats_before_filter['total_1hop']}")
    print(f"  Total Two-Hop Paths: {stats_before_filter['total_2hop']}")
    print(f"  Overall Total Paths: {stats_before_filter['total_1hop'] + stats_before_filter['total_2hop']}")
    
    print("\nFiltered Path Counts:")
    print(f"  Total One-Hop Paths: {stats_after_filter['total_1hop']}")
    print(f"  Total Two-Hop Paths: {stats_after_filter['total_2hop']}")
    print(f"  Overall Total Paths: {stats_after_filter['total_1hop'] + stats_after_filter['total_2hop']}")

    deleted_2hop = stats_before_filter['total_2hop'] - stats_after_filter['total_2hop']
    print(f"\nNumber of Two-Hop Paths Deleted: {deleted_2hop}")
    if stats_before_filter['total_2hop'] > 0:
        print(f"  Percentage of Two-Hop Paths Deleted: {deleted_2hop / stats_before_filter['total_2hop']:.2%}")
    
    print(f"\nNumber of OOD CUIs (in CUI map but not in dataset facts): {num_ood_cuis}")

    if case_threshold_scores:
        print("\nBM25 Score Thresholds for Retained 2-Hop Paths (per case where 2-hops were kept):")
        print(f"  Min Threshold: {min(case_threshold_scores):.4f}")
        print(f"  Max Threshold: {max(case_threshold_scores):.4f}")
        print(f"  Mean Threshold: {np.mean(case_threshold_scores):.4f}")
        print(f"  Median Threshold: {np.median(case_threshold_scores):.4f}")
    else:
        print("\nNo 2-hop paths were retained based on BM25 scores (or no 2-hop paths existed).")

    print("\n🎉 BM25 filtering process complete!")

if __name__ == '__main__':
    
    
    main()