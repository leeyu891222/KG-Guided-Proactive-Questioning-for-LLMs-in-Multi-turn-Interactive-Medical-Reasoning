import json
import pickle
import networkx as nx
import spacy
import scispacy
from scispacy.linking import EntityLinker
from tqdm import tqdm
import argparse
import os
from typing import List, Set, Dict, Any

def load_resources(annotations_path: str, mediq_path: str, graph_path: str):
    """加載所有必要的資源文件。"""
    print("--- 開始加載資源 ---")
    
    # 1. 加載現有的標註文件
    print(f"正在加載標註文件: {annotations_path}")
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations_data = json.load(f)
    
    # 2. 加載 MEDIQ 資料集並轉換為以 id 為鍵的字典
    print(f"正在加載 MEDIQ 資料集: {mediq_path}")
    mediq_data_dict = {}
    with open(mediq_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # 將 id 轉為字串以匹配 annotations.json 的鍵
            mediq_data_dict[str(item['id'])] = item
            
    # 3. 加載知識圖譜
    print(f"正在加載知識圖譜: {graph_path}")
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    
    graph_cui_set = set(graph.nodes())
    print(f"資源加載完畢。標註案例數: {len(annotations_data)}, MEDIQ案例數: {len(mediq_data_dict)}, 圖譜節點數: {len(graph_cui_set)}")
    return annotations_data, mediq_data_dict, graph, graph_cui_set

def initialize_scispacy() -> spacy.Language:
    """初始化 ScispaCy 模型。"""
    print("正在初始化 ScispaCy 模型...")
    nlp = spacy.load("en_core_sci_md")
    linker = EntityLinker(resolve_abbreviations=True, name="umls")
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    print("ScispaCy 初始化成功。")
    return nlp

def extract_valid_cuis(text: str, nlp: spacy.Language, graph_cuis: Set[str]) -> List[str]:
    """根據規則從文本中提取有效的 CUIs。"""
    if not text:
        return []
    doc = nlp(text)
    valid_cuis = []
    for ent in doc.ents:
        if ent._.kb_ents:
            for cui_candidate, _ in ent._.kb_ents:
                if cui_candidate in graph_cuis:
                    valid_cuis.append(cui_candidate)
                    break # 找到第一個就跳出
    return list(set(valid_cuis)) # 返回唯一的CUI

def find_paths_up_to_2_hops(graph: nx.DiGraph, source_cuis: List[str], target_cuis: List[str]) -> List[List[str]]:
    """在圖中查找從源CUI到目標CUI的2跳內路徑，並避免找到自環路徑。"""
    if not source_cuis or not target_cuis:
        return []
    
    all_found_paths = []
    original_target_set = set(target_cuis)

    for source in source_cuis:
        if source not in graph:
            continue
        
        # --- 【修改點 A】: 避免 source 和 target 相同 ---
        # 創建一個不包含當前 source 的臨時目標集合。
        # 這樣可以從根本上避免找到 A->A 這樣終點就是起點的路徑。
        currentTarget_set = original_target_set - {source}
        
        # 如果排除掉 source 後目標集合為空，則跳過此 source 的搜索
        if not currentTarget_set:
            continue

        # 1-hop paths
        for neighbor in graph.neighbors(source):
            # 使用臨時目標集合進行檢查
            if neighbor in currentTarget_set:
                relation = graph.get_edge_data(source, neighbor).get('label', 'related_to')
                all_found_paths.append([source, relation, neighbor])
        
        # 2-hop paths
        for neighbor1 in graph.neighbors(source):
            if neighbor1 not in graph:
                continue
            
            # --- 【修改點 B】: 避免中間節點就是源節點 (例如 A->A->B) ---
            if neighbor1 == source:
                continue

            relation1 = graph.get_edge_data(source, neighbor1).get('label', 'related_to')
            for neighbor2 in graph.neighbors(neighbor1):
                # --- 【修改點 C】: 避免路徑在第二步回到源節點 (例如 A->B->A) ---
                if neighbor2 == source:
                    continue
                
                # 使用臨時目標集合進行檢查
                if neighbor2 in currentTarget_set:
                    relation2 = graph.get_edge_data(neighbor1, neighbor2).get('label', 'related_to')
                    all_found_paths.append([source, relation1, neighbor1, relation2, neighbor2])
                    
    return all_found_paths

def main(args):
    # 1. 加載所有資源
    annotations_data, mediq_data, graph, graph_cui_set = load_resources(
        args.annotations_file, args.mediq_file, args.graph_file
    )
    nlp = initialize_scispacy()
    
    print("\n--- 開始擴充標註資料 ---")
    # 2. 遍歷每個案例進行擴充
    for case_id, data in tqdm(annotations_data.items(), desc="正在擴充案例"):
        if case_id not in mediq_data:
            tqdm.write(f"警告：在 MEDIQ 資料中找不到案例 ID: {case_id}，已跳過。")
            continue
        
        case_qa_data = mediq_data[case_id]
        
        # 3. 添加 question 和 answer 文本
        data['question'] = case_qa_data.get('question', '')
        data['answer'] = case_qa_data.get('answer', '')
        
        # 4. 抽取 question 和 answer 的 CUI
        data['question_cui'] = extract_valid_cuis(data['question'], nlp, graph_cui_set)
        data['answer_cui'] = extract_valid_cuis(data['answer'], nlp, graph_cui_set)
        
        # 5. 標註 Q-Fact 和 Fact-A 的路徑
        paths_qa = {}
        fact_cuis_list = data.get('facts_cuis', [])
        
        # 標註 Q -> Fact 路徑
        for fact_id, fact_cuis in enumerate(fact_cuis_list):
            q_to_f_paths = find_paths_up_to_2_hops(graph, data['question_cui'], fact_cuis)
            if q_to_f_paths:
                paths_qa[f"Q_{fact_id}"] = q_to_f_paths
        
        # 標註 Fact -> A 路徑
        for fact_id, fact_cuis in enumerate(fact_cuis_list):
            f_to_a_paths = find_paths_up_to_2_hops(graph, fact_cuis, data['answer_cui'])
            if f_to_a_paths:
                paths_qa[f"{fact_id}_A"] = f_to_a_paths
        
        data['paths_between_QA_facts'] = paths_qa

    # 6. 保存擴充後的資料
    print(f"\n--- 擴充完畢，正在保存到文件: {args.output_file} ---")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations_data, f, indent=2, ensure_ascii=False)
        
    print("所有操作已成功完成！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Augment training annotations with Q&A data from MEDIQ dataset.")
    parser.add_argument('--annotations_file', type=str, default="MediQ/mediq_test_annotations_bm25_20.json", help='Path to the existing JSON annotations file.')
    parser.add_argument('--mediq_file', type=str, default="MediQ/all_test_convo.jsonl", help='Path to the MEDIQ JSONL dataset file.')
    parser.add_argument('--graph_file', type=str, default="drknows/SNOMED_CUI_MAJID_Graph_wSelf.pkl", help='Path to the NetworkX graph pickle file.')
    parser.add_argument('--output_file', type=str, default="MediQ/mediq_test_annotations_wQA.json", help='Path to save the new augmented JSON file.')
    
    args = parser.parse_args()
    main(args)