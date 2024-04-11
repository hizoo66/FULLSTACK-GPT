import streamlit as st

st.title("QuizGPT")

st.markdown("""
Welcome!
            
Use this chatbot to ask questions to an AI about your files!
            
Upload your files on the sidebar.
""")

with st.sidebar:
    file = st.file_uploader("Upload a .txt .pdf or .docs file", type=["pdf","txt","docx"])
    openai_key = st.text_input("Please put your OpenAI_key")
    button = st.button("save key")
    github_url = st.text("https://github.com/hizoo66/FULLSTACK-GPT/blob/main/pages/01_DocumentGPT.py")
    app_url = st.text("")
    maker = st.text("made by Hizoo")

    if button:
        save_api_key(openai_key)
        st.write(f"API_KEY = {openai_key}")
        if openai_key == "":
            st.warning("CAN'T RECOGNIZED OPEN_API_KEY")

if openai_key:
    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ],
        openai_api_key=st.session_state["openai_key"]
    )
else:
    st.markdown("PLEASE PUT YOUR OPENAI_API_KEY")