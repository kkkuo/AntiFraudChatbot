#HuggingFace models
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
#Langchain models
from langchain.chains import ConversationalRetrievalChain, StuffDocumentsChain, LLMChain #
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

llm = None
retriever = None
prompt = None
chat = None

def load_model():
    global llm, retriever, prompt, chat
    if chat is None:
        llm = ChatGoogleGenerativeAI(
            model = 'gemini-2.0-flash',
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature = 0.1
        )
        embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        faiss_db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True) #如果沒有加上allow_dangerous_deserialization會不能用之前已經跑完下載好的.pkl檔案
        retriever = faiss_db.as_retriever(search_kwargs={"k": 3})
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        prompt_qgen = PromptTemplate.from_template("""
                "請根據對話紀錄和使用者的追問，產生一個可獨立理解的問題。\n"
                "對話紀錄：{chat_history}\n"
                "使用者問題：{question}\n"
                "獨立問題：
                                                   """)
        question_generator_chain = LLMChain(llm=llm, prompt=prompt_qgen)

        prompt_qa = PromptTemplate.from_template("""
                    你是台灣反詐騙專家。請根據資料庫案例直接判斷使用者情況。

                    使用者問題：{question}
                    資料庫案例：{context}

                    如果資料庫有相關詐騙案例，請用繁體中文簡潔回答：
                    「你遇到的情況就是詐騙，[簡述相關案例內容]。因此，請停止與對方聯絡。」

                    如果沒有相關案例，請用繁體中文簡潔回答：
                    「目前沒有相關案例，但請保持警惕，如有疑慮請撥打165專線。」

                    """)
        #用 context + 問題產生回答
        combine_docs_chain = StuffDocumentsChain(
            llm_chain=LLMChain(llm=llm, prompt=prompt_qa),
            document_variable_name="context"
        ) 
        
        chat = ConversationalRetrievalChain(
            combine_docs_chain = combine_docs_chain,
            retriever=retriever,
            question_generator = question_generator_chain,
            memory = memory
        )
        #param response_if_no_docs_found: str | None = None If specified, the chain will return a fixed response if no docs are found for the question.

        return chat

