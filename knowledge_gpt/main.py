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

# Initialize session state if it doesn't exist
if 'processed' not in st.session_state:
    st.session_state['processed'] = False

if 'queried' not in st.session_state:
    st.session_state['queried'] = False

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
    accept_multiple_files=True,
    help="Scanned documents are not supported yet!",
)

model: str = st.selectbox("Model", options=MODEL_LIST)  # type: ignore

with st.expander("Advanced Options"):
    return_all_chunks = st.checkbox("Show all chunks retrieved from vector search")
    show_full_doc = st.checkbox("Show parsed contents of the document")

if not uploaded_files:
    st.stop()

folder_indices = []
for uploaded_file in uploaded_files:
    # ... code to process each file ...
    # For now, let's just add a pass statement if there's no actual processing.
    pass

# Set processed to True once the document is processed
st.session_state['processed'] = True

def handle_query(folder_indices, query, return_all_chunks, llm, uploaded_files):
    # Create a list of document options, adding an "All documents" option at the start
    document_options = ["All documents"] + [f"Document {i}" for i, _ in enumerate(uploaded_files, start=1)]
    selected_document = st.selectbox("Select document", options=document_options)

    # Output Columns
    answer_col, sources_col = st.columns(2)

    if selected_document == "All documents":
        # Query all documents
        all_results = []
        for folder_index in folder_indices:
            result = query_folder(
                folder_index=folder_index,
                query=query,
                return_all=return_all_chunks,
                llm=llm,
            )
            all_results.append(result)
        # ... handle/display results for all documents
    else:
        # Query the selected document
        folder_index = folder_indices[document_options.index(selected_document) - 1]  # Adjusted index due to "All documents" option
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

with st.form(key="qa_form"):
    query = st.text_area("Ask a question about the document")
    submit = st.form_submit_button("Submit")

    if submit:
        if not is_query_valid(query):
            st.stop()
        llm = get_llm(model=model, openai_api_key=openai_api_key, temperature=0)
        handle_query(folder_indices, query, return_all_chunks, llm, uploaded_files)

if not uploaded_files:
    st.stop()

if show_full_doc:
    with st.expander("Document"):
        # Hack to get around st.markdown rendering LaTeX
        st.markdown(f"<p>{wrap_doc_in_html(file.docs)}</p>", unsafe_allow_html=True)
