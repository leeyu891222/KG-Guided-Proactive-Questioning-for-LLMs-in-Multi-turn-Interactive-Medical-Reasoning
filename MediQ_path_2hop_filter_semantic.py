import json
import pickle
import random
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# --- Configuration ---
ANNOTATION_FILE_PATH = "./MediQ/mediq_test_annotations.json"  # Replace with your input annotation file
MEDIQ_FILE_PATH = "./MediQ/all_test_convo.jsonl"            # Replace with your MEDIQ data file
CUI_TO_TEXT_PKL_PATH = "./drknows/sm_t047_cui_aui_eng.pkl"   # Replace with your CUI mapping PKL file
FILTERED_ANNOTATION_OUTPUT_PATH = "./MediQ/mediq_test_annotations_semantic_k2.json"
SAPBERT_MODEL_NAME = 'pritamdeka/S-BioBert-snli-multinli-stsb' # or your local path
TOP_K_TWO_HOP_PATHS = 2 # Keep the top K best 2-hop paths for each fact pair
WEIGHT_SEGMENT1 = 0.3
WEIGHT_SEGMENT2 = 0.3
WEIGHT_MEDIQ_QUESTION = 0.4

# --- Helper Functions ---

def get_cui_text_representation(cui, cui_to_text_map):
    """Gets the first preferred text for a CUI."""
    if cui in cui_to_text_map and cui_to_text_map[cui]:
        return cui_to_text_map[cui][0][1] # Return the first AUI's text
    return cui # Fallback to CUI itself if no text found

def convert_path_segment_to_text(cui1, rel, cui2, cui_to_text_map):
    """Converts a path segment (CUI-Rel-CUI) to a text string."""
    text1 = get_cui_text_representation(cui1, cui_to_text_map)
    text2 = get_cui_text_representation(cui2, cui_to_text_map)
    return f"{text1} {rel} {text2}"

def convert_full_two_hop_path_to_text(cui_start, rel1, cui_mid, rel2, cui_end, cui_to_text_map):
    """Converts a full two-hop path to a text string."""
    text_start = get_cui_text_representation(cui_start, cui_to_text_map)
    text_mid = get_cui_text_representation(cui_mid, cui_to_text_map)
    text_end = get_cui_text_representation(cui_end, cui_to_text_map)
    # Simple concatenation, can be made more natural if needed
    return f"{text_start} {rel1} {text_mid}, which {rel2} {text_end}"

def calculate_similarity(emb1, emb2):
    """Calculates cosine similarity between two embeddings."""
    if emb1 is None or emb2 is None or emb1.ndim == 0 or emb2.ndim == 0:
        return 0.0
    # Ensure embeddings are 2D for cosine_similarity
    emb1_2d = emb1.reshape(1, -1) if emb1.ndim == 1 else emb1
    emb2_2d = emb2.reshape(1, -1) if emb2.ndim == 1 else emb2
    return cosine_similarity(emb1_2d, emb2_2d)[0][0]

# --- Main Script ---

def main():
    print("🚀 Starting GT Path Filtering Process...")

    # 1. Load Data
    print("\n--- 📂 Loading Data ---")
    try:
        with open(ANNOTATION_FILE_PATH, 'r', encoding='utf-8') as f:
            annotations_data = json.load(f)
        print(f"✅ Loaded annotations from: {ANNOTATION_FILE_PATH}")

        mediq_data_map = {}
        with open(MEDIQ_FILE_PATH, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                try:
                    item = json.loads(line) # Parse each line
                    # Ensure 'id' key exists for mapping
                    if 'id' in item:
                        mediq_data_map[item['id']] = item
                    else:
                        print(f"⚠️ Warning: MEDIQ data line {line_number} missing 'id' key. Skipping line: {line.strip()}")
                except json.JSONDecodeError as e_line:
                    print(f"❌ ERROR parsing MEDIQ data line {line_number}: {e_line}. Line content: {line.strip()}")
                    # Decide if you want to stop or continue
                    # return # Stop on first error
                    continue # Skip problematic line and continue
            print(f"✅ Loaded MEDIQ data (line by line) from: {MEDIQ_FILE_PATH}")
            if not mediq_data_map:
                print(f"⚠️ Warning: MEDIQ data map is empty after loading. Check file content and 'id' keys.")

        with open(CUI_TO_TEXT_PKL_PATH, 'rb') as f:
            cui_to_text_map = pickle.load(f)
        print(f"✅ Loaded CUI-to-text mappings from: {CUI_TO_TEXT_PKL_PATH}")

    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not find a required file. Please check paths. Missing: {e.filename}")
        return
    except Exception as e:
        print(f"❌ ERROR loading data: {e}")
        return

    # 2. Initialize Sentence Embedding Model
    print("\n--- 🧠 Initializing Sentence Embedding Model (SapBERT) ---")
    try:
        embedding_model = SentenceTransformer(SAPBERT_MODEL_NAME)
        print(f"✅ SapBERT model '{SAPBERT_MODEL_NAME}' initialized.")
    except Exception as e:
        print(f"❌ ERROR initializing SapBERT model: {e}")
        print("Ensure you have the sentence-transformers library installed and the model name/path is correct.")
        return

    filtered_annotations_data = {}
    stats_before = {"total_paths": 0, "one_hop_paths": 0, "two_hop_paths": 0}
    stats_after = {"total_paths": 0, "one_hop_paths": 0, "two_hop_paths": 0}

    print("\n--- 🔍 Processing and Filtering Paths ---")
    for case_id, case_content in tqdm(annotations_data.items(), desc="Processing Cases"):
        filtered_case_content = {
            "atomic_facts": case_content.get("atomic_facts", []),
            "facts_cuis": case_content.get("facts_cuis", []),
            "paths_between_facts": {}
        }

        # Get MEDIQ question for this case
        # The MEDIQ ID might be an integer, ensure case_id from annotation file matches
        mediq_case_id_type_adjusted = int(case_id) if case_id.isdigit() else case_id # Adjust if MEDIQ IDs are int
        
        current_mediq_item = mediq_data_map.get(mediq_case_id_type_adjusted)
        if not current_mediq_item:
            # print(f"⚠️ Warning: No MEDIQ question found for case_id '{case_id}'. Skipping MEDIQ question relevance for this case.")
            emb_mediq_question = None
        else:
            mediq_question_text = current_mediq_item.get("question", "")
            if mediq_question_text:
                 emb_mediq_question = embedding_model.encode([mediq_question_text])[0]
            else:
                # print(f"⚠️ Warning: MEDIQ question text is empty for case_id '{case_id}'.")
                emb_mediq_question = None
        
        original_paths_for_case = case_content.get("paths_between_facts", {})
        for path_key, path_list in original_paths_for_case.items():
            stats_before["total_paths"] += len(path_list)
            
            try:
                idx_str_list = path_key.split('_')
                if len(idx_str_list) != 2:
                    # print(f"Warning: Invalid path_key format '{path_key}' in case '{case_id}'. Skipping.")
                    continue
                known_fact_idx, unknown_fact_idx = int(idx_str_list[0]), int(idx_str_list[1])
            except ValueError:
                # print(f"Warning: Could not parse indices from path_key '{path_key}' in case '{case_id}'. Skipping.")
                continue

            # Get known and unknown fact texts and their embeddings
            atomic_facts = case_content.get("atomic_facts", [])
            if not (0 <= known_fact_idx < len(atomic_facts) and 0 <= unknown_fact_idx < len(atomic_facts)):
                # print(f"Warning: Fact indices out of bounds for path_key '{path_key}' in case '{case_id}'. Skipping.")
                continue
                
            known_fact_text = atomic_facts[known_fact_idx]
            unknown_fact_text = atomic_facts[unknown_fact_idx]
            emb_known_fact = embedding_model.encode([known_fact_text])[0]
            emb_unknown_fact = embedding_model.encode([unknown_fact_text])[0]

            kept_paths_for_key = []
            two_hop_paths_with_scores = []

            for path_data in path_list:
                if not isinstance(path_data, list) or not path_data:
                    continue

                path_len_nodes = sum(1 for item in path_data if isinstance(item, str) and item.startswith('C'))
                num_relations = len(path_data) - path_len_nodes
                
                is_one_hop = (path_len_nodes == 2 and num_relations == 1 and len(path_data) == 3)
                is_two_hop = (path_len_nodes == 3 and num_relations == 2 and len(path_data) == 5)

                if is_one_hop:
                    stats_before["one_hop_paths"] += 1
                    kept_paths_for_key.append(path_data) # Keep all 1-hop paths
                    stats_after["one_hop_paths"] += 1
                elif is_two_hop:
                    stats_before["two_hop_paths"] += 1
                    ck, r1, cm, r2, cu = path_data

                    # Convert segments to text
                    seg1_text = convert_path_segment_to_text(ck, r1, cm, cui_to_text_map)
                    seg2_text = convert_path_segment_to_text(cm, r2, cu, cui_to_text_map)
                    full_path_text = convert_full_two_hop_path_to_text(ck, r1, cm, r2, cu, cui_to_text_map)

                    # Get embeddings
                    emb_seg1 = embedding_model.encode([seg1_text])[0]
                    emb_seg2 = embedding_model.encode([seg2_text])[0]
                    emb_full_path = embedding_model.encode([full_path_text])[0]

                    # Calculate relevance scores
                    score1 = calculate_similarity(emb_seg1, emb_known_fact)
                    score2 = calculate_similarity(emb_seg2, emb_unknown_fact)
                    score3 = calculate_similarity(emb_full_path, emb_mediq_question) if emb_mediq_question is not None else 0.0

                    total_score = (WEIGHT_SEGMENT1 * score1 +
                                   WEIGHT_SEGMENT2 * score2 +
                                   WEIGHT_MEDIQ_QUESTION * score3)
                    two_hop_paths_with_scores.append((path_data, total_score))
                # else: path_data has an unexpected format/length

            # Select top-k for two-hop paths
            if two_hop_paths_with_scores:
                two_hop_paths_with_scores.sort(key=lambda x: x[1], reverse=True)
                for path_data, score in two_hop_paths_with_scores[:TOP_K_TWO_HOP_PATHS]:
                    kept_paths_for_key.append(path_data)
                    stats_after["two_hop_paths"] += 1
            
            if kept_paths_for_key:
                filtered_case_content["paths_between_facts"][path_key] = kept_paths_for_key
                stats_after["total_paths"] += len(kept_paths_for_key)

        filtered_annotations_data[case_id] = filtered_case_content

    # 4. Save Filtered Annotations
    print("\n--- 💾 Saving Filtered Annotations ---")
    try:
        with open(FILTERED_ANNOTATION_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(filtered_annotations_data, f, indent=2)
        print(f"✅ Filtered annotations saved to: {FILTERED_ANNOTATION_OUTPUT_PATH}")
    except Exception as e:
        print(f"❌ ERROR saving filtered annotations: {e}")

    # 5. Print Statistics
    print("\n--- 📊 Statistics ---")
    print("Before Filtering:")
    print(f"  Total Paths: {stats_before['total_paths']}")
    print(f"  One-Hop Paths: {stats_before['one_hop_paths']}")
    print(f"  Two-Hop Paths: {stats_before['two_hop_paths']}")
    print("\nAfter Filtering:")
    print(f"  Total Paths: {stats_after['total_paths']}")
    print(f"  One-Hop Paths: {stats_after['one_hop_paths']} (should be same as before or less if original files had errors)")
    print(f"  Two-Hop Paths: {stats_after['two_hop_paths']}")

    deleted_two_hop = stats_before['two_hop_paths'] - stats_after['two_hop_paths']
    total_deleted = stats_before['total_paths'] - stats_after['total_paths']
    print(f"\n  Number of Two-Hop Paths Deleted: {deleted_two_hop}")
    print(f"  Total Paths Deleted: {total_deleted}")
    if stats_before['total_paths'] > 0:
        print(f"  Percentage of Paths Deleted: {total_deleted / stats_before['total_paths']:.2%}")
    if stats_before['two_hop_paths'] > 0 :
        print(f"  Percentage of Two-Hop Paths Deleted: {deleted_two_hop / stats_before['two_hop_paths']:.2%}")

    print("\n🎉 Filtering process complete!")

if __name__ == '__main__':

        
    main()