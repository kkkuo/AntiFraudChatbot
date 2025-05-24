import gradio as gr
from scripts.model import load_model

chat = load_model()

def chat_function(message, history):
    try:
        # 呼叫你的 ConversationalRetrievalChain
        response = chat({"question": message})
        return response["answer"]
    except Exception as e:
        return f"發生錯誤：{str(e)}"
    
# 建立聊天介面
iface = gr.ChatInterface(
    fn=chat_function,
    title="反詐騙諮詢機器人",
    description="請描述您遇到的情況，我會幫您判斷是否為詐騙。",
    examples=["有人要我投資虛擬貨幣", "接到陌生來電說我中獎了"],
)

if __name__ == "__main__":
    iface.launch(share=True)