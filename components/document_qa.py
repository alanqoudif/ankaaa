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
    
    # Add direct speech-to-text functionality
    if language == "English":
        speech_button_text = "🎤 Click to Speak"
        speech_active_text = "🔴 Listening... Click to Stop"
        speech_placeholder = "Listening to your question..."
    else:  # Arabic
        speech_button_text = "🎤 انقر للتحدث"
        speech_active_text = "🔴 جاري الاستماع... انقر للإيقاف"
        speech_placeholder = "جاري الاستماع إلى سؤالك..."
    
    # Initialize session state for speech recording status
    if "qa_speech_active" not in st.session_state:
        st.session_state.qa_speech_active = False
    
    # Create a button for speech input
    speech_col1, speech_col2 = st.columns([1, 3])
    
    with speech_col1:
        button_text = speech_active_text if st.session_state.qa_speech_active else speech_button_text
        button_type = "primary" if st.session_state.qa_speech_active else "secondary"
        
        if st.button(button_text, key="qa_speech_button", type=button_type):
            # Toggle speech recording state
            st.session_state.qa_speech_active = not st.session_state.qa_speech_active
            st.rerun()
    
    # Add JavaScript for speech recognition
    if st.session_state.qa_speech_active:
        with speech_col2:
            st.write(speech_placeholder)
            
        # Determine the language code for speech recognition
        speech_lang = "en-US" if language == "English" else "ar-EG"
        
        # Create JavaScript for speech recognition
        js_code = f"""
        <script>
        const speechDiv = window.parent.document.querySelector('[data-testid="stMarkdown"] p');
        const textArea = window.parent.document.querySelector('textarea');
        const button = window.parent.document.querySelector('[data-testid="baseButton-secondary"]');
        
        if (typeof window.speechRecognition === 'undefined') {{
            window.speechRecognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            window.speechRecognition.continuous = true;
            window.speechRecognition.interimResults = true;
            window.speechRecognition.lang = '{speech_lang}';
            
            window.speechRecognition.onresult = (event) => {{
                let finalTranscript = '';
                for (let i = event.resultIndex; i < event.results.length; ++i) {{
                    if (event.results[i].isFinal) {{
                        finalTranscript += event.results[i][0].transcript;
                    }}
                }}
                
                if (finalTranscript !== '') {{
                    textArea.value = finalTranscript;
                    // Trigger input event to update Streamlit state
                    const event = new Event('input', {{ bubbles: true }});
                    textArea.dispatchEvent(event);
                }}
            }};
            
            window.speechRecognition.onerror = (event) => {{
                console.error('Speech recognition error', event.error);
                button.click(); // Stop recording on error
            }};
        }}
        
        // Start recognition
        try {{
            window.speechRecognition.start();
        }} catch (e) {{
            console.error('Could not start speech recognition', e);
        }}
        </script>
        """
        st.components.v1.html(js_code, height=0)
    
    # Voice input through file upload as fallback option
    st.write(upload_text)
    
    with st.expander("Upload Audio File"):
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
                    
                    # Improve the readability of the text using OpenAI
                    original_content = doc.page_content
                    
                    # Show loading spinner while improving text
                    with st.spinner("Improving text readability..." if language == "English" else "تحسين قراءة النص..."):
                        improved_content = llm_manager.improve_legal_text_readability(original_content, language)
                    
                    # Format the original content with proper styling
                    original_html = f"""
                    <details>
                        <summary style="cursor: pointer; color: #555; font-weight: bold; margin-bottom: 5px;">
                            {"Show Original Text" if language == "English" else "عرض النص الأصلي"}
                        </summary>
                        <div dir="auto" style="background-color: #f0f0f0; padding: 10px; 
                                              border-radius: 5px; margin: 10px 0; 
                                              font-family: 'Arial', sans-serif; line-height: 1.5;">
                            {original_content}
                        </div>
                    </details>
                    """
                    
                    # Format the improved content with proper styling
                    improved_html = f"""
                    <div dir="auto" style="background-color: #f0f0f0; padding: 10px; 
                                          border-radius: 5px; margin: 10px 0; 
                                          font-family: 'Arial', sans-serif; line-height: 1.5;
                                          border-left: 4px solid #2c3e50;">
                        {improved_content}
                    </div>
                    """
                    
                    # Show the improved text by default with option to view original
                    st.markdown(f"<h6>{'Improved Text:' if language == 'English' else 'النص المحسن:'}</h6>", unsafe_allow_html=True)
                    st.markdown(improved_html, unsafe_allow_html=True)
                    st.markdown(original_html, unsafe_allow_html=True)
                    st.markdown("---")
