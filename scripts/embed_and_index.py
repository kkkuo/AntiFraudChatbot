import pandas as pd
from pathlib import Path
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def embed_and_save_faiss_langchain():
    # 載入資料
    data_path = Path("data/fraud_data.csv")
    df = pd.read_csv(data_path)

    df.fillna("", inplace=True)

    # 準備文本（合併標題與摘要）
    texts = (df['CaseTitle'] + '。' + df['Summary']).tolist()

    # 建立 LangChain 嵌入器（用相同模型）
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # 建立 FAISS 向量資料庫
    vectorstore = FAISS.from_texts(texts, embedding=embedding_model)

    # 儲存 FAISS index
    index_dir = Path("faiss_index")
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))

    print(f"✅ 已用 LangChain 建立並儲存 {len(texts)} 筆資料的 FAISS index")

if __name__ == "__main__":
    embed_and_save_faiss_langchain()
