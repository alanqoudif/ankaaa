import streamlit as st
from utils.llm_manager import LLMManager
from utils.speech_to_text import SpeechToText, preprocess_audio

def law_comparison(vector_store, available_laws, language):
    """
    Component for comparing different laws.
    
    Args:
        vector_store: Vector store with document embeddings
        available_laws: List of available law names
        language: Language to use for interface (English or Arabic)
    """
    # Initialize LLM manager and speech-to-text
    llm_manager = LLMManager()
    stt = SpeechToText("en" if language == "English" else "ar")
    
    # Set up the UI based on language
    if language == "English":
        st.header("Law Comparison")
        first_law_label = "First Law"
        second_law_label = "Second Law"
        text_input_label = "Comparison Query (or use voice input)"
        voice_button_text = "Record Voice"
        stop_button_text = "Stop Recording"
        compare_button_text = "Compare Laws"
        loading_text = "Comparing laws..."
        voice_instruction = "Click to speak your comparison query"
        transcribing_text = "Transcribing audio..."
        voice_placeholder = "Your spoken query will appear here..."
        no_laws_text = "Please select two different laws to compare."
        query_placeholder = "How do these laws differ in their approach to penalties?"
    else:  # Arabic
        st.header("مقارنة القوانين")
        first_law_label = "القانون الأول"
        second_law_label = "القانون الثاني"
        text_input_label = "استعلام المقارنة (أو استخدم إدخال الصوت)"
        voice_button_text = "تسجيل الصوت"
        stop_button_text = "إيقاف التسجيل"
        compare_button_text = "مقارنة القوانين"
        loading_text = "جاري مقارنة القوانين..."
        voice_instruction = "انقر للتحدث بطلب المقارنة الخاص بك"
        transcribing_text = "جاري نسخ الصوت..."
        voice_placeholder = "ستظهر استفسارك المنطوق هنا..."
        no_laws_text = "يرجى تحديد قانونين مختلفين للمقارنة."
        query_placeholder = "كيف تختلف هذه القوانين في نهجها تجاه العقوبات؟"
    
    # Select laws to compare
    col1, col2 = st.columns(2)
    with col1:
        first_law = st.selectbox(first_law_label, available_laws, key="first_law")
    with col2:
        # Filter out the first selected law
        second_law_options = [law for law in available_laws if law != first_law]
        second_law = st.selectbox(second_law_label, second_law_options, key="second_law")
    
    # Initialize session state for voice input
    if "voice_input" not in st.session_state:
        st.session_state.voice_input = ""
    if "recording" not in st.session_state:
        st.session_state.recording = False
    
    # Text input for comparison query
    query = st.text_area(text_input_label, value=st.session_state.voice_input, 
                         placeholder=query_placeholder, height=100)
    
    # Voice input section
    st.write(voice_instruction)
    
    voice_col1, voice_col2 = st.columns(2)
    
    with voice_col1:
        if not st.session_state.recording:
            if st.button(voice_button_text):
                st.session_state.recording = True
                st.rerun()
        else:
            if st.button(stop_button_text, type="primary"):
                # Stop recording and process audio
                st.session_state.recording = False
                
                # Get the audio data
                if "audio_data" in st.session_state:
                    with st.spinner(transcribing_text):
                        # Process the audio
                        processed_audio = preprocess_audio(st.session_state.audio_data)
                        
                        # Transcribe the audio
                        transcription = stt.transcribe_audio(processed_audio)
                        
                        # Update the text input with transcription
                        st.session_state.voice_input = transcription
                
                st.rerun()
    
    # Display audio recorder when recording
    if st.session_state.recording:
        audio_data = st.audio_recorder(pause_threshold=2.0, sample_rate=16000)
        if audio_data is not None:
            st.session_state.audio_data = audio_data
            st.text(voice_placeholder)
    
    # Process comparison
    if st.button(compare_button_text):
        if first_law == second_law:
            st.warning(no_laws_text)
            return
        
        with st.spinner(loading_text):
            # Extract content for the first law
            first_law_docs = [doc for doc in vector_store.documents 
                             if doc.metadata.get("law_name") == first_law]
            first_law_content = "\n".join([doc.page_content for doc in first_law_docs])
            
            # Extract content for the second law
            second_law_docs = [doc for doc in vector_store.documents 
                              if doc.metadata.get("law_name") == second_law]
            second_law_content = "\n".join([doc.page_content for doc in second_law_docs])
            
            # If the content is too long, use a summarized version
            max_content_length = 8000  # Adjust based on LLM context limits
            
            if len(first_law_content) > max_content_length:
                first_law_content = first_law_content[:max_content_length] + "..."
            
            if len(second_law_content) > max_content_length:
                second_law_content = second_law_content[:max_content_length] + "..."
            
            # Generate comparison
            comparison = llm_manager.compare_laws(
                first_law, first_law_content,
                second_law, second_law_content,
                language
            )
            
            # Display comparison
            st.write("### Comparison Results")
            st.write(comparison)
