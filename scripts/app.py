import gradio as gr
from model import load_model

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
    title="反詐聊天機器人",
    description="請描述您遇到的情況，我會幫您判斷是否為詐騙。",
    examples=["我在臉書上看到投資老師叫我加line好友，教我穩賺不賠的方法", "有人跟我買東西，說我給的賣貨便連結點進去有問題"],
)

if __name__ == "__main__":
    iface.launch(share=True)