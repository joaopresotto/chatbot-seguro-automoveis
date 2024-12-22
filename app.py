import ollama
import uuid
import torch
import streamlit as st
from src.chatbot_seguros import ChatbotSeguros

st.title("Chatbot Seguros - Santander, Bradesco, Porto Seguro e Suhai")

# initialize history
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    
# initialize model
chatbot = ChatbotSeguros()
session_id = str(uuid.uuid4())

#Display chat messages from history on app rerun
# for message in st.session_state["messages"]:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

if prompt := st.chat_input("Insira sua mensagem.."):
    # add latest message to history in format {role, content}
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message = st.write(chatbot.gerar_resposta(prompt, session_id))
        st.session_state["messages"].append({"role": "assistant", "content": message})