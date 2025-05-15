from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
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
    
    PROMPT_TEMPLATE = """你是一位專業的反詐騙諮詢助手。使用者會告訴你他遇到的情況，並詢問你是否可能遇到詐騙。請根據以下背景資料，清楚地回答使用者的問題。
    如果找到相關詐騙資訊，請告訴使用者「這很有可能是詐騙」，並提供一則具體的詐騙案例或相關資訊。請務必提供具體的詐騙案例或相關資訊，讓使用者能夠理解為什麼這是詐騙。
    如果使用者還是認為這不是詐騙，請再提供一則具體的詐騙案例或相關資訊。請務必提供具體的詐騙案例或相關資訊，並建議使用者撥打165反詐騙專線。
    如果使用者詢問的問題與詐騙無關，請告訴使用者「這不是詐騙」，並提供具體的理由或建議。請務必提供具體的理由或建議，讓使用者能夠理解為什麼這不是詐騙。
    如果找不到相關資訊，請建議使用者撥打165反詐騙專線，並且告訴使用者「這很有可能是詐騙」。

    背景資料：
    {context}

    使用者問題：
    {question}

    請用繁體中文回答，語氣親切且專業。
    如果有人詢問與詐騙無關的問題，請回覆：「請你去找別人聊天，不要佔用公共資源。」
    """

    PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE
    )

    # 建立 QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain

def run_qa(query):
    qa_chain = load_qa_chain()
    result = qa_chain.invoke(query)
    print(result["result"])

    print("相關資料來源：")
    for doc in result["source_documents"]:
        print(doc.page_content)
        print("-" * 40)

# 測試用
if __name__ == "__main__":
    test_query = "有人說我帳戶異常，要我操作ATM，我該怎麼辦？"
    run_qa(test_query)
