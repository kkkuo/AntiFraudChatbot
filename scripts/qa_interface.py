from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def load_qa_chain():
    # 模型名稱
    model_id = "ziqingyang/chinese-alpaca-2-7b"

    # 載入 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto"
    )

    # 建立 text-generation pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, do_sample=True)

    # 包裝成 LangChain 的 LLM
    llm = HuggingFacePipeline(pipeline=pipe)

    # 建立向量嵌入器
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # 載入 FAISS 資料庫（假設你已儲存在 faiss_index/）
    vectorstore = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)

    # 建立 QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    return qa_chain

def run_qa(query):
    qa_chain = load_qa_chain()
    result = qa_chain.invoke(query)

    print("🔎 問題：", query)
    print("\n📘 回答：", result["result"])
    print("\n📎 相關資料來源：")
    for doc in result["source_documents"]:
        print(doc.page_content)
        print("-" * 40)

# 測試用
if __name__ == "__main__":
    test_query = "有人說我帳戶異常，要我操作ATM，我該怎麼辦？"
    run_qa(test_query)
