from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def load_vectorstore():
    # 載入 LangChain 的 HuggingFace Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # 從本地載入 FAISS 向量資料庫
    vectorstore = FAISS.load_local("faiss_index", embedding=embedding_model)
    
    return vectorstore

def search(query, k=3):
    # 載入向量資料庫
    vectorstore = load_vectorstore()

    # 執行相似度搜尋
    results = vectorstore.similarity_search(query, k=k)

    # 輸出搜尋結果
    for i, res in enumerate(results):
        print(f"第 {i+1} 筆結果：")
        print(res.page_content)  # 顯示最相似的文本
        print("-" * 40)

if __name__ == "__main__":
    # 提示用戶輸入查詢文本
    user_query = input("請輸入你的查詢：")
    
    # 執行搜尋並顯示結果
    search(user_query)
