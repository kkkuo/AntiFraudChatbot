import pandas as pd
from pathlib import Path
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def embed_and_save_faiss_langchain():
    data_path = Path("data/fraud_data.csv")
    df = pd.read_csv(data_path)

    df.fillna("", inplace=True)

    #Save metadata
    documents = []
    for index, row in df.iterrows():
        content_to_embed = str(row['Summary'])
        metadata = {
            'CaseId': str(row['Id']),
            'Casetitle': str(row['CaseTitle']),
            'Date': str(row['CaseDate'])
        }
        if content_to_embed.strip():
            documents.append(Document(page_content = content_to_embed, metadata = metadata))

    #Chunking
    spliter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        separators=["\n\n", "。", "！", "？", "\n", "，"]
    )
    
    splited_docs = spliter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    vectorstore = FAISS.from_documents(splited_docs, embedding=embedding_model)

    index_dir = Path("faiss_index")
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))

    print(f"已儲存 {len(documents)} 筆資料的 FAISS index")

if __name__ == "__main__":
    embed_and_save_faiss_langchain()
