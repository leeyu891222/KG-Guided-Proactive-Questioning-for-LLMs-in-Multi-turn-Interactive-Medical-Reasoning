import torch
import pickle
import networkx as nx
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

def generate_relation_embeddings():
    """
    加載知識圖譜，提取所有唯一的關係名稱，
    並使用 SapBERT 為其生成語義嵌入，最後保存為 pickle 檔案。
    """
    
    # --- 1. 設定文件路徑和模型名稱 ---
    # 請確保此腳本執行時，相對路徑是正確的
    GRAPH_NX_FILE = "./drknows/SNOMED_CUI_MAJID_Graph_wSelf.pkl"
    SAPBERT_MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    OUTPUT_FILE = "relation_sapbert_embeddings.pkl"
    
    # 檢查圖譜文件是否存在
    if not os.path.exists(GRAPH_NX_FILE):
        print(f"錯誤：找不到知識圖譜文件 -> {GRAPH_NX_FILE}")
        print("請確保路徑正確，或將文件放置在正確的位置。")
        return

    # --- 2. 加載知識圖譜並提取唯一關係 ---
    print(f"正在從 {GRAPH_NX_FILE} 加載知識圖譜...")
    try:
        g_nx = pickle.load(open(GRAPH_NX_FILE, "rb"))
        print("知識圖譜加載成功。")
    except Exception as e:
        print(f"加載圖譜時出錯: {e}")
        return

    print("正在提取唯一的關係名稱...")
    # 使用集合（set）來自動處理重複的關係名稱
    unique_relations = set()
    for _, _, edge_data in g_nx.edges(data=True):
        if 'label' in edge_data and edge_data['label']:
            unique_relations.add(edge_data['label'])
            
    relation_names = sorted(list(unique_relations))
    
    if not relation_names:
        print("警告：在圖譜中沒有找到任何帶有 'label' 的關係。無法生成嵌入。")
        return
        
    print(f"共找到 {len(relation_names)} 種唯一的關係。")
    print(f"部分關係範例: {relation_names[:5]}")

    # --- 3. 加載 SapBERT 模型和 Tokenizer ---
    print(f"正在加載 SapBERT 模型: {SAPBERT_MODEL_NAME}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(SAPBERT_MODEL_NAME)
    model = AutoModel.from_pretrained(SAPBERT_MODEL_NAME).to(device)
    model.eval() # 設置為評估模式
    print(f"模型已加載到 {device}。")

    # --- 4. 為每個關係生成嵌入 ---
    print("開始生成關係嵌入...")
    relation_embeddings = {}
    
    with torch.no_grad(): # 在無梯度的模式下進行，以節省記憶體和加速
        for name in tqdm(relation_names, desc="Generating Embeddings"):
            # 使用 tokenizer 將關係名稱轉換為模型輸入格式
            inputs = tokenizer(name, return_tensors="pt", truncation=True, padding=True).to(device)
            
            # 獲取模型的輸出
            outputs = model(**inputs)
            
            # 我們使用 pooler_output，這是 [CLS] token 對應的、代表整個句子語義的嵌入
            embedding = outputs.pooler_output
            
            # 將嵌入向量從 GPU 移至 CPU，並轉換為 numpy 陣列以便序列化
            relation_embeddings[name] = embedding.detach().cpu().numpy()

    print("所有關係的嵌入已生成。")

    # --- 5. 保存結果到 pickle 檔案 ---
    print(f"正在將結果保存到 {OUTPUT_FILE}...")
    try:
        with open(OUTPUT_FILE, "wb") as f_out:
            pickle.dump(relation_embeddings, f_out)
        print(f"成功！嵌入檔案已保存至 {os.path.abspath(OUTPUT_FILE)}")
    except Exception as e:
        print(f"保存檔案時出錯: {e}")

if __name__ == '__main__':
    generate_relation_embeddings()