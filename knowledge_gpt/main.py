import streamlit as st

from knowledge_gpt.ui import (
    wrap_doc_in_html,
    is_query_valid,
    is_file_valid,
    is_open_ai_key_valid,
    display_file_read_error,
)

from knowledge_gpt.core.caching import bootstrap_caching

from knowledge_gpt.core.parsing import read_file
from knowledge_gpt.core.chunking import chunk_file
from knowledge_gpt.core.embedding import embed_files
from knowledge_gpt.core.qa import query_folder
from knowledge_gpt.core.utils import get_llm

EMBEDDING = "openai"
VECTOR_STORE = "faiss"
MODEL_LIST = ["gpt-3.5-turbo", "gpt-4"]

st.set_page_config(page_title="HCD-Helper", layout="wide")
st.header("HCD-Helper")

# Enable caching for expensive functions
bootstrap_caching()

openai_api_key = st.text_input(
    "Enter your OpenAI API key. You can get a key at "
    "[https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)"
)

uploaded_files = st.file_uploader(
    "Upload pdf, docx, or txt files",
    type=["pdf", "docx", "txt"],
    help="Scanned documents are not supported yet!",
    accept_multiple_files=True  # Allows multiple files
)

model: str = st.selectbox("Model", options=MODEL_LIST)  # type: ignore

with st.expander("Advanced Options"):
    return_all_chunks = st.checkbox("Show all chunks retrieved from vector search")
    show_full_doc = st.checkbox("Show parsed contents of the document")

if not uploaded_files:
    st.stop()

folder_indices = []
for uploaded_file in uploaded_files:
    try:
        file = read_file(uploaded_file)
    except Exception as e:
        display_file_read_error(e, file_name=uploaded_file.name)
        continue  # Skip to the next file if there's an error

    chunked_file = chunk_file(file, chunk_size=300, chunk_overlap=0)

    if not is_file_valid(file):
        continue  # Skip to the next file if not valid

    if not is_open_ai_key_valid(openai_api_key, model):
        continue  # Skip to the next file if API key is not valid

    with st.spinner(f"Indexing document {uploaded_file.name}... This may take a while⏳"):
        folder_index = embed_files(
            files=[chunked_file],
            embedding=EMBEDDING if model != "debug" else "debug",
            vector_store=VECTOR_STORE if model != "debug" else "debug",
            openai_api_key=openai_api_key,
        )
        folder_indices.append(folder_index)  # Keep track of indices

    if show_full_doc:
        with st.expander(f"Document {uploaded_file.name}"):
            st.markdown(f"<p>{wrap_doc_in_html(file.docs)}</p>", unsafe_allow_html=True)

with st.form(key="qa_form"):
    query = st.text_area("Ask a question about the documents")
    submit = st.form_submit_button("Submit")

if submit:
    if not is_query_valid(query):
        st.stop()

    # Output Columns
    answer_col, sources_col = st.columns(2)

    llm = get_llm(model=model, openai_api_key=openai_api_key, temperature=0)
    
    # Assume the query is for the first document; adjust as needed.
    folder_index = folder_indices[0]
    result = query_folder(
        folder_index=folder_index,
        query=query,
        return_all=return_all_chunks,
        llm=llm,
    )

    with answer_col:
        st.markdown("#### Answer")
        st.markdown(result.answer)

    with sources_col:
        st.markdown("#### Sources")
        for source in result.sources:
            st.markdown(source.page_content)
            st.markdown(source.metadata["source"])
            st.markdown("---")
