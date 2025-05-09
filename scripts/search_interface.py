# scripts/search_interface.py

import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

def search_similar_cases(query, top_k=3):
    # 載入模型
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # 編碼查詢句子
    query_vec = model.encode([query])

    # 載入 FAISS index
    index = faiss.read_index("faiss_index/index_file.index")

    # 載入 labels（你可以從 data/final.csv 再度讀入）
    df = pd.read_csv("data/final.csv")

    # 做搜尋
    D, I = index.search(np.array(query_vec), k=top_k)

    # 顯示搜尋結果
    for idx in I[0]:
        print("相似案例：", df.iloc[idx]['Summary'])
        print("詐騙類型：", df.iloc[idx]['CaseTitle'])
        print("-" * 30)
