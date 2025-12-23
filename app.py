import streamlit as st
from chatbot import ask_question

st.set_page_config(page_title="Dynamic Knowledge Chatbot")

st.title("Dynamic Knowledge Base Chatbot")

query = st.text_input("Ask a question:")

if st.button("Ask"):
    answer = ask_question(query)
    st.write(answer)