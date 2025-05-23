#HuggingFace models
from langchain.llms import HuggingFacePipeline #之後要改成langchain_huggingface
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
#Langchain models
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.memory import ConversationBufferMemory


llm = None
retriever = None
prompt = None
chat = None

def load_model():
    global llm, retriever, prompt, chat
    if chat is None:
        model_id = 'Qwen/Qwen3-0.6B'
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto"
        )
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, framework = 'pt', max_new_tokens=512, do_sample=True) #max_new_tokens:限制模型回答的長度; do_sample:決定生成的文字是否會隨機取樣，False的話模型每次都只會選擇機率最高的字，回答較死板。Temperature決定的是隨機的程度
        prompt = PromptTemplate.from_template("""
                你是一位專業的反詐騙諮詢助手。使用者會告訴你他遇到的情況，請根據提供的背景資料和對話歷史，清楚地回答使用者遇到的情況是不是詐騙。
                當你在回答使用者的時候，請附上一則罪相關的案例，並告知使用者「如果還有疑慮，請撥打165專線諮詢專員」。
                如果使用者問你詐騙以外的內容，在每句回覆後面加上與使用者傳入的內容最相關的詐騙案件。
                如果找不到相關案例，而使用者很懷疑這是詐騙，告訴使用者撥打165專線詢問專員。
                只能使用繁體中文回覆。                                                           
                Question : {question}
                Context : {context}
                Answer :                                                            
                                            """)
        llm = HuggingFacePipeline(pipeline=pipe)
        
        embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        faiss_db = FAISS.load_local("faiss_index/index.faiss", embedding_model, allow_dangerous_deserialization=True) #如果沒有加上allow_dangerous_deserialization會不能用之前已經跑完下載好的.pkl檔案
        retriever = faiss_db.as_retriever(search_kwargs={"k": 3})
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        chat = ConversationalRetrievalChain(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        return chat

