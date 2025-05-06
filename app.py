import os
import streamlit as st
from pathlib import Path
import glob

# Import components
from components.document_qa import document_qa
from components.article_summarizer import article_summarizer
from components.law_comparison import law_comparison
from components.document_creator import document_creator
from components.case_analyzer import case_analyzer

# Import utilities
from utils.document_processor import process_pdfs, get_available_laws
from utils.vector_store import VectorStore

# Set page config
st.set_page_config(
    page_title="Omani Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
)

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = False
if "available_laws" not in st.session_state:
    st.session_state.available_laws = []
if "language" not in st.session_state:
    st.session_state.language = "English"

# Title and description
st.title("Omani Legal AI Assistant")

# Language selection
language_options = ["English", "Arabic"]
language_col1, language_col2 = st.columns([1, 5])
with language_col1:
    selected_language = st.selectbox(
        "Language", 
        language_options, 
        index=language_options.index(st.session_state.language)
    )
    
    if selected_language != st.session_state.language:
        st.session_state.language = selected_language
        st.rerun()

# Display content based on selected language
if st.session_state.language == "English":
    st.write("Your comprehensive assistant for Omani legal matters.")
    process_btn_text = "Process Legal Documents"
    processing_text = "Processing documents... This may take a few minutes."
    no_docs_text = "No PDF documents found in the current directory. Please add PDF files to continue."
else:  # Arabic
    st.write("مساعدك الشامل للمسائل القانونية العمانية.")
    process_btn_text = "معالجة الوثائق القانونية"
    processing_text = "جاري معالجة المستندات... قد يستغرق ذلك بضع دقائق."
    no_docs_text = "لم يتم العثور على مستندات PDF في الدليل الحالي. يرجى إضافة ملفات PDF للمتابعة."

# Find PDF files in the current directory
pdf_files = glob.glob("*.pdf") + glob.glob("**/*.pdf", recursive=True)

# Process PDF files if available
if pdf_files:
    if not st.session_state.processed_docs:
        if st.button(process_btn_text):
            with st.spinner(processing_text):
                # Process PDFs and create vector store
                documents = process_pdfs(pdf_files)
                st.session_state.vector_store = VectorStore(documents)
                st.session_state.available_laws = get_available_laws(documents)
                st.session_state.processed_docs = True
                st.rerun()
else:
    st.warning(no_docs_text)
    st.stop()

# Only show features if documents are processed
if st.session_state.processed_docs:
    # Main navigation tabs
    if st.session_state.language == "English":
        tabs = st.tabs(["Document Q&A", "Article Summarizer", "Law Comparison", "Document Creator", "Case Analyzer"])
    else:  # Arabic
        tabs = st.tabs(["البحث في الوثائق", "ملخص المادة", "مقارنة القوانين", "إنشاء المستندات", "تحليل الحالة"])
    
    # Document Q&A
    with tabs[0]:
        document_qa(st.session_state.vector_store, st.session_state.language)
    
    # Article Summarizer
    with tabs[1]:
        article_summarizer(st.session_state.vector_store, st.session_state.available_laws, st.session_state.language)
    
    # Law Comparison
    with tabs[2]:
        law_comparison(st.session_state.vector_store, st.session_state.available_laws, st.session_state.language)
    
    # Document Creator
    with tabs[3]:
        document_creator(st.session_state.vector_store, st.session_state.language)
    
    # Case Analyzer
    with tabs[4]:
        case_analyzer(st.session_state.vector_store, st.session_state.language)

# Footer
st.markdown("---")
if st.session_state.language == "English":
    st.markdown("Omani Legal AI Assistant © 2023 | Powered by Gemini and LangChain")
else:  # Arabic
    st.markdown("مساعد الذكاء الاصطناعي القانوني العماني © 2023 | مدعوم من جيميني و LangChain")
