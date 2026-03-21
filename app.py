import streamlit as st
import os
from langchain_community.embeddings import HuggingFaceEmbeddings


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

st.set_page_config(
    page_title="DocAnalyse AI",
    page_icon="🌸",
    layout="wide"
)

st.markdown("""
<style>

/* ===== FORCE WHITE TEXT EVERYWHERE ===== */
html, body, .stApp, [class*="css"] {
    color: #FFFFFF !important;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF !important;
}

/* ===== FILE UPLOADER ===== */
[data-testid="stFileUploader"] * {
    color: #FFFFFF !important;
}

/* ===== DROPDOWN / SELECT ===== */
.stSelectbox * {
    color: #FFFFFF !important;
}

/* ===== SUCCESS / INFO MESSAGES ===== */
.stAlert * {
    color: #FFFFFF !important;
}

/* ===== CHAT MESSAGES ===== */
[data-testid="stChatMessage"] * {
    color: #FFFFFF !important;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* ===== BUTTON TEXT ===== */
button {
    color: #FFFFFF !important;
}

/* ===== NORMAL INPUTS ===== */
input, textarea {
    color: #FFFFFF !important;
}

/* ===== CHAT INPUT BAR (BOTTOM) ===== */
/* Make background dark so white text is visible */

[data-testid="stChatInput"] textarea {
    background-color: #1E1E1E !important;
    color: #FFFFFF !important;
}

/* Placeholder text */
[data-testid="stChatInput"] textarea::placeholder {
    color: #BBBBBB !important;
}

/* Chat input container */
[data-testid="stChatInput"] {
    background-color: #1E1E1E !important;
    border-radius: 12px !important;
}

/* ===== EXPANDER (Sources) ===== */
[data-testid="stExpander"] * {
    color: #FFFFFF !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
# 🌸 Docanalyse
### Document analyser tool 
                Upload documents and learn from them
""")



with st.sidebar:

    st.title("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    st.markdown("---")

    st.markdown("### ⚙️ Analysis Mode")

    mode = st.selectbox(
        "Choose mode",
        ["General Q&A", "Executive Summary", "Action Items"]
    )

    st.markdown("---")
    st.caption("Built for quick analysis")


if uploaded_files:

    all_docs = []

    for uploaded_file in uploaded_files:

        file_path = f"temp_{uploaded_file.name}"

        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = uploaded_file.name

        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)


    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=""
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        return_source_documents=True
    )

    st.success(f"✅ {len(uploaded_files)} documents ready")



    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])



    user_prompt = st.chat_input("Ask about your documents…")

    if user_prompt:

        st.session_state.messages.append(
            {"role": "user", "content": user_prompt}
        )

        st.chat_message("user").write(user_prompt)

        if mode == "Executive Summary":
            query = "Provide an executive summary: " + user_prompt
        elif mode == "Action Items":
            query = "Extract action items and decisions: " + user_prompt
        else:
            query = user_prompt

        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents..."):
                result = qa({"query": query})
                answer = result["result"]
                st.write(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )


        with st.expander("📚 Sources"):

            for doc in result["source_documents"]:
                st.write(
                    f"📄 **{doc.metadata.get('source')}** — Page {doc.metadata.get('page', 'N/A')}"
                )
                st.write(doc.page_content[:300] + "...")
                st.write("---")