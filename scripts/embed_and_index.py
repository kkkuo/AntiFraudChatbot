import torch
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

def embed_and_save_faiss():
    # 載入資料
    data_path = Path("data/fraud_data.csv")
    df = pd.read_csv(data_path)

    # 準備文本
    texts = (df['CaseTitle'] + '。' + df['Summary']).tolist()

    # 載入模型
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # 檢查是否有 CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 產生嵌入向量
    embeddings = model.encode(texts, show_progress_bar=True, device=device)

    # 建立 FAISS 索引
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # 儲存 FAISS index
    index_dir = Path("faiss_index")
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "fraud_cases.index"))

    print(f"✅ 已建立並儲存 {len(texts)} 筆資料的 FAISS index")

if __name__ == "__main__":
    embed_and_save_faiss()
