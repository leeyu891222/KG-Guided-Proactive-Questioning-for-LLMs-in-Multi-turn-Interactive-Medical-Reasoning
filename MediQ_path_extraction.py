# Step 0: Import libraries and load necessary models/data
import json
import pickle
import networkx as nx
import spacy
import scispacy # noqa F401
from scispacy.linking import EntityLinker
import time
from collections import defaultdict

print("載入資源中...")

# --- 配置 ---
# !!! 請確保這些路徑是正確的 !!!
MEDIQA_FILE = './MediQ/all_dev_convo.jsonl'
GRAPH_FILE = './drknows/SNOMED_CUI_MAJID_Graph_wSelf.pkl'
# CUI_DICT_FILE = './drknows/sm_t047_cui_aui_eng.pkl' # 暫不直接用於過濾，主要用圖譜節點
OUTPUT_ANNOTATION_FILE = './MediQ/mediq_dev_annotations.json' # 輸出的標註文件

# SciSpaCy 模型 - 你可能需要根據安裝情況調整模型名稱
# 例如：en_core_sci_sm, en_core_sci_md, en_core_sci_lg, en_ner_bc5cdr_md 等
# 確保模型包含實體識別能力
SCISPACY_MODEL_NAME = "en_core_sci_md"

# --- 載入圖譜 ---
try:
    with open(GRAPH_FILE, 'rb') as f:
        drknows_digraph = pickle.load(f, encoding='latin1')
    print(f"成功載入圖譜: {GRAPH_FILE}")
    # 創建圖譜節點 CUI 的集合，以便快速查找
    graph_nodes_set = set(drknows_digraph.nodes())
    print(f"圖譜包含 {len(graph_nodes_set):,} 個節點 (CUIs)。")
except FileNotFoundError:
    print(f"錯誤：找不到圖譜文件 {GRAPH_FILE}。請檢查路徑。")
    exit()
except Exception as e:
    print(f"載入圖譜時發生錯誤: {e}")
    exit()

# --- 載入 CUI 字典 (可選，主要用於參考) ---
# try:
#     with open(CUI_DICT_FILE, 'rb') as f:
#         drknows_vocab = pickle.load(f, encoding='latin1')
#     print(f"成功載入 CUI 字典: {CUI_DICT_FILE}")
# except FileNotFoundError:
#     print(f"警告：找不到 CUI 字典文件 {CUI_DICT_FILE}。")
#     drknows_vocab = {}
# except Exception as e:
#     print(f"載入 CUI 字典時發生錯誤: {e}")
#     drknows_vocab = {}

# --- 載入 scispaCy 模型和 UMLS 鏈接器 ---
try:
    print(f"載入 scispaCy 模型: {SCISPACY_MODEL_NAME}...")
    nlp = spacy.load(SCISPACY_MODEL_NAME)
    print("模型載入完成。")
    # 添加 Entity Linker - 假設已安裝並配置好 UMLS 索引
    # 索引的創建和載入可能需要額外步驟，參考 scispaCy 文件
    print("添加 UMLS 實體鏈接器...")
    linker = EntityLinker(resolve_abbreviations=True, name="umls")
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    print("鏈接器添加完成。(注意：確保 UMLS 索引可用)")
except OSError:
     print(f"錯誤：無法載入 scispaCy 模型 '{SCISPACY_MODEL_NAME}'。請確保已下載模型 (e.g., python -m spacy download en_core_sci_lg) 或模型名稱正確。")
     exit()
except Exception as e:
    print(f"載入 scispaCy 或鏈接器時發生錯誤: {e}")
    # 根據錯誤，可能需要創建或指定 UMLS 索引路徑
    print("請檢查 scispaCy 和 UMLS 鏈接器的安裝與配置。")
    exit()

print("資源載入完成。")

# -----------------------------------------------------------------------------
#  步驟 1: CUI 抽取與驗證函數
# -----------------------------------------------------------------------------
def extract_and_validate_cuis(text, graph_nodes):
    """
    從文本中抽取實體，鏈接到 UMLS CUI，並只保留圖譜中存在的 CUI。
    Args:
        text (str): 需要處理的 atomic_fact 文本。
        graph_nodes (set): 包含知識圖譜中所有節點 CUI 的集合。
    Returns:
        list: 該文本對應的、且存在於圖譜中的 CUI 列表 (去重)。
    """
    validated_cuis = set()
    doc = nlp(text)
    entities = doc.ents

    # print(f"  處理文本: '{text}'")
    # print(f"  找到實體: {[ent.text for ent in entities]}")

    for entity in entities:
        # 每個實體可能有多個 UMLS 鏈接候選
        # ent._.kb_ents 是一個 (cui, score) 的列表，按分數排序
        linked_entities = entity._.kb_ents
        # print(f"    實體 '{entity.text}' 的候選 CUIs: {linked_entities}")
        if linked_entities:
            # 優先選擇分數最高且存在於圖譜中的 CUI
            best_cui_found = None
            for cui, score in linked_entities:
                if cui in graph_nodes:
                    best_cui_found = cui
                    # print(f"      選擇 CUI: {cui} (分數: {score:.4f}, 存在於圖譜中)")
                    break # 找到第一個（分數最高）就停止
            if best_cui_found:
                 validated_cuis.add(best_cui_found)
            # else:
            #     print(f"      實體 '{entity.text}' 的所有候選 CUI 都不在圖譜中。")

    return list(validated_cuis)

# -----------------------------------------------------------------------------
#  步驟 2: Path 抽取函數 (最多 2 跳)
# -----------------------------------------------------------------------------
def find_paths_between_sets(graph, set1_cuis, set2_cuis):
    """
    查找兩個 CUI 集合之間所有長度為 1 或 2 跳的路徑。
    Args:
        graph (nx.DiGraph): 知識圖譜 (NetworkX DiGraph)。
        set1_cuis (list): 起始 CUI 集合。
        set2_cuis (list): 目標 CUI 集合。
    Returns:
        list: 包含所有找到的路徑的列表。
              路徑格式: [cui1] (1跳), [cui1, rel1, cui2] (1跳帶關係),
                       [cui1, rel1, cui_mid, rel2, cui2] (2跳帶關係)
    """
    paths = []
    if not set1_cuis or not set2_cuis:
        return paths

    for u in set1_cuis:
        for v in set2_cuis:
            if u == v: continue # 不查找自己到自己的路徑

            # 查找 1 跳路徑
            if graph.has_edge(u, v):
                try:
                    label = graph.edges[u, v].get('label', 'REL_1') # 獲取關係標籤，若無則用預設
                    paths.append([u, label, v])
                except KeyError: # 處理可能的邊屬性缺失
                     paths.append([u, 'REL_1', v])


            # 查找 2 跳路徑
            # 使用 NetworkX 的 neighbors 或 successors (對於有向圖)
            try:
                 # neighbors() 在無向圖視角下工作，successors() 嚴格按有向圖
                 # 根據圖譜性質選擇，這裡假設是有向圖
                for mid_node in graph.successors(u):
                    if mid_node == v or mid_node == u: continue # 避免 A->B->B 或 A->A->B

                    if graph.has_edge(mid_node, v):
                        try:
                           label1 = graph.edges[u, mid_node].get('label', 'REL_A')
                           label2 = graph.edges[mid_node, v].get('label', 'REL_B')
                           paths.append([u, label1, mid_node, label2, v])
                        except KeyError: # 處理可能的邊屬性缺失
                            # 如果要記錄不帶標籤的路徑，可以這樣處理
                            # paths.append([u, 'REL_A', mid_node, 'REL_B', v])
                            pass # 或者忽略缺少標籤的路徑

            except nx.NetworkXError: # 處理節點不在圖中的罕見情況 (理論上已被過濾)
                pass
            except Exception as e: # 捕捉其他可能的圖操作錯誤
                # print(f"查找2跳路徑時出錯 u={u}, v={v}: {e}")
                pass


    return paths

# -----------------------------------------------------------------------------
#  主處理流程
# -----------------------------------------------------------------------------
all_annotations = {}
start_time = time.time()
processed_count = 0

print(f"\n開始處理 MediQ 文件: {MEDIQA_FILE}")

try:
    with open(MEDIQA_FILE, 'r', encoding='utf-8') as f_in:
        for i, line in enumerate(f_in):
            if i % 50 == 0 and i > 0: # 每處理 50 筆輸出一次進度
                elapsed_time = time.time() - start_time
                print(f"已處理 {i} 筆案例... ({elapsed_time:.2f} 秒)")

            try:
                record = json.loads(line.strip())
                case_id = record.get('id')
                atomic_facts = record.get('atomic_facts', [])

                if case_id is None or not atomic_facts:
                    # print(f"警告：第 {i+1} 行記錄缺少 'id' 或 'atomic_facts'，跳過。")
                    continue

                # --- 步驟 1: 對每個 fact 抽取並驗證 CUI ---
                facts_cuis = []
                for fact_text in atomic_facts:
                    validated_cuis = extract_and_validate_cuis(fact_text, graph_nodes_set)
                    facts_cuis.append(validated_cuis)
                    # print(f"  Fact '{fact_text[:50]}...' -> Validated CUIs: {validated_cuis}")


                # --- 步驟 2: 查找所有 fact 對之間的 2 跳路徑 ---
                case_paths = {}
                num_facts = len(facts_cuis)
                for idx1 in range(num_facts):
                    for idx2 in range(num_facts):
                        if idx1 == idx2: continue # 不比較自己和自己

                        set1 = facts_cuis[idx1]
                        set2 = facts_cuis[idx2]

                        if set1 and set2: # 只有當兩個 fact 都有有效 CUI 時才查找路徑
                            paths_found = find_paths_between_sets(drknows_digraph, set1, set2)
                            if paths_found: # 只記錄找到路徑的 fact 對
                                case_paths[f"{idx1}_{idx2}"] = paths_found # 使用 "idx1_idx2" 作為 key

                all_annotations[case_id] = {
                    "atomic_facts": atomic_facts, # 保留原始 facts 文本
                    "facts_cuis": facts_cuis,     # 每個 fact 對應的驗證後 CUI 列表
                    "paths_between_facts": case_paths # fact 對之間的 2 跳路徑
                }
                processed_count += 1

            except json.JSONDecodeError:
                print(f"警告：第 {i+1} 行 JSON 解析錯誤，跳過。")
            except Exception as e:
                print(f"警告：處理第 {i+1} 行記錄時發生錯誤: {e}，跳過。")
                # print(f"錯誤記錄內容: {line[:200]}...") # 打印部分錯誤行內容幫助排查

except FileNotFoundError:
    print(f"錯誤：找不到 MediQ 文件 {MEDIQA_FILE}。請檢查路徑。")
    exit()
except Exception as e:
    print(f"讀取 MediQ 文件時發生錯誤: {e}")
    exit()

end_time = time.time()
print(f"\n處理完成！共處理 {processed_count} 筆有效案例。")
print(f"總耗時: {(end_time - start_time):.2f} 秒。")

# --- 儲存標註結果 ---
print(f"正在將標註結果儲存到: {OUTPUT_ANNOTATION_FILE}")
try:
    with open(OUTPUT_ANNOTATION_FILE, 'w', encoding='utf-8') as f_out:
        # 使用 indent=2 讓 JSON 文件更易讀，但會增加文件大小
        json.dump(all_annotations, f_out, ensure_ascii=False, indent=2)
    print("儲存成功！")
except Exception as e:
    print(f"儲存標註文件時發生錯誤: {e}")