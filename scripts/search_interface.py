from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.load_local("faiss_index", embedding=embedding_model)
    return vectorstore

def search(query, k=3):
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search(query, k=k)

    for i, res in enumerate(results):
        print(f"第 {i+1} 筆結果：")
        print(res.page_content)
        print("-" * 40)

if __name__ == "__main__":
    user_query = input("請輸入你的查詢：")
    search(user_query)
