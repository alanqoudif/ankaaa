import streamlit as st
from utils.openai_manager import OpenAIManager
from utils.speech_to_text import SpeechToText, preprocess_audio
import io
import time

def document_qa(vector_store, language):
    """
    Component for document Q&A functionality.
    
    Args:
        vector_store: Vector store with document embeddings
        language: Language to use for interface (English or Arabic)
    """
    # Initialize OpenAI manager and speech-to-text
    llm_manager = OpenAIManager()
    stt = SpeechToText("en" if language == "English" else "ar")
    
    # Set up the UI based on language
    if language == "English":
        st.header("Ask Questions about Omani Laws")
        query_placeholder = "What is the penalty for theft under Omani law?"
        button_text = "Ask Question"
        loading_text = "Searching legal documents..."
        no_results_text = "No relevant information found in the legal documents. Please try rephrasing your question."
        upload_text = "Or upload an audio file with your question"
        transcribing_text = "Transcribing audio..."
    else:  # Arabic
        st.header("اسأل عن القوانين العمانية")
        query_placeholder = "ما هي عقوبة السرقة بموجب القانون العماني؟"
        button_text = "اسأل السؤال"
        loading_text = "جاري البحث في الوثائق القانونية..."
        no_results_text = "لم يتم العثور على معلومات ذات صلة في الوثائق القانونية. يرجى إعادة صياغة سؤالك."
        upload_text = "أو قم بتحميل ملف صوتي يحتوي على سؤالك"
        transcribing_text = "جاري نسخ الصوت..."
    
    # Query input
    if "qa_voice_input" not in st.session_state:
        st.session_state.qa_voice_input = ""
    
    query = st.text_area("", value=st.session_state.qa_voice_input, placeholder=query_placeholder, height=100)
    
    # Voice input through file upload
    st.write(upload_text)
    audio_file = st.file_uploader("", type=["wav", "mp3", "ogg"], key="qa_audio_upload")
    
    if audio_file is not None and "qa_audio_processed" not in st.session_state:
        with st.spinner(transcribing_text):
            # Read the audio file
            audio_bytes = audio_file.read()
            
            # Process the audio
            processed_audio = preprocess_audio(audio_bytes)
            
            # Transcribe the audio
            transcription = stt.transcribe_audio(processed_audio)
            
            # Update the query
            st.session_state.qa_voice_input = transcription
            st.session_state.qa_audio_processed = True
            
            # Rerun to update the text area with the transcription
            st.rerun()
    
    # Clear the processed flag if there's no audio file
    if audio_file is None and "qa_audio_processed" in st.session_state:
        st.session_state.pop("qa_audio_processed")
    
    # Process query
    if st.button(button_text):
        if not query:
            return
        
        with st.spinner(loading_text):
            # Search for relevant documents
            search_results = vector_store.search(query, k=5)
            
            if not search_results:
                st.warning(no_results_text)
                return
            
            # Gather context from search results
            context = ""
            for doc, score in search_results:
                # Only include relevant results (adjust threshold if needed)
                if score > 0.2:
                    # Add metadata about the source
                    law_name = doc.metadata.get("law_name", "Unknown Law")
                    source = doc.metadata.get("source", "Unknown Source")
                    
                    # Add to context
                    context += f"\nFrom {law_name} ({source}):\n{doc.page_content}\n"
            
            # Use the LLM to answer based on context
            answer = llm_manager.answer_legal_question(query, context, language)
            
            # Display the answer
            st.write("### Answer")
            st.write(answer)
            
            # Option to show sources
            with st.expander("Show Sources" if language == "English" else "عرض المصادر"):
                for i, (doc, score) in enumerate(search_results):
                    law_name = doc.metadata.get("law_name", "Unknown Law")
                    source = doc.metadata.get("source", "Unknown Source")
                    page_num = doc.metadata.get("page", "Unknown")
                    
                    # Create styled header for source
                    if language == "English":
                        header = f"**Source {i+1}: {law_name}**"
                        file_info = f"*File: {source} | Page: {page_num} | Relevance: {score:.2f}*"
                    else:
                        header = f"**المصدر {i+1}: {law_name}**"
                        file_info = f"*الملف: {source} | الصفحة: {page_num} | الصلة: {score:.2f}*"
                    
                    st.markdown(header)
                    st.markdown(file_info)
                    
                    # Display content in a more readable format
                    st.markdown("---")
                    st.markdown("##### Content:")
                    
                    # Format the content with proper styling
                    content_html = f"""
                    <div dir="auto" style="background-color: #f0f0f0; padding: 10px; 
                                          border-radius: 5px; margin: 10px 0; 
                                          font-family: 'Arial', sans-serif; line-height: 1.5;">
                        {doc.page_content}
                    </div>
                    """
                    st.markdown(content_html, unsafe_allow_html=True)
                    st.markdown("---")
