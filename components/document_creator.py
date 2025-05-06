import streamlit as st
from utils.llm_manager import LLMManager
from utils.pdf_generator import PDFGenerator
from utils.image_generator import ImageGenerator
from utils.speech_to_text import SpeechToText, preprocess_audio
import tempfile
import os
import datetime

def document_creator(vector_store, language):
    """
    Component for creating legal documents.
    
    Args:
        vector_store: Vector store with document embeddings
        language: Language to use for interface (English or Arabic)
    """
    # Initialize LLM manager, PDF generator, image generator, and speech-to-text
    llm_manager = LLMManager()
    pdf_gen = PDFGenerator(language)
    img_gen = ImageGenerator(language)
    stt = SpeechToText("en" if language == "English" else "ar")
    
    # Set up the UI based on language
    if language == "English":
        st.header("Document Creator")
        doc_type_label = "Document Type"
        doc_types = ["Employment Contract", "Power of Attorney", "Non-Disclosure Agreement", 
                    "Lease Agreement", "Sale Contract", "Legal Notice", "Custom Document"]
        spec_input_label = "Document Specifications (or use voice input)"
        voice_button_text = "Record Voice"
        stop_button_text = "Stop Recording"
        create_button_text = "Create Document"
        loading_text = "Generating document..."
        voice_instruction = "Click to speak your document specifications"
        transcribing_text = "Transcribing audio..."
        voice_placeholder = "Your spoken specifications will appear here..."
        pdf_download_text = "Download PDF"
        image_download_text = "Download Image"
        spec_placeholder = """
        Please describe the details for the document, such as:
        - Names of the parties involved
        - Key terms and conditions
        - Dates, locations, and any other relevant information
        
        Example: "Create an employment contract between XYZ Company and John Doe for the position of Legal Consultant, with a monthly salary of 1,500 OMR, starting on January 1, 2023, for a duration of 2 years."
        """
    else:  # Arabic
        st.header("إنشاء المستندات")
        doc_type_label = "نوع المستند"
        doc_types = ["عقد عمل", "توكيل رسمي", "اتفاقية عدم إفشاء", 
                    "عقد إيجار", "عقد بيع", "إشعار قانوني", "مستند مخصص"]
        spec_input_label = "مواصفات المستند (أو استخدم إدخال الصوت)"
        voice_button_text = "تسجيل الصوت"
        stop_button_text = "إيقاف التسجيل"
        create_button_text = "إنشاء المستند"
        loading_text = "جاري إنشاء المستند..."
        voice_instruction = "انقر للتحدث بمواصفات المستند الخاص بك"
        transcribing_text = "جاري نسخ الصوت..."
        voice_placeholder = "ستظهر المواصفات المنطوقة هنا..."
        pdf_download_text = "تنزيل PDF"
        image_download_text = "تنزيل الصورة"
        spec_placeholder = """
        يرجى وصف تفاصيل المستند، مثل:
        - أسماء الأطراف المعنية
        - الشروط والأحكام الرئيسية
        - التواريخ والمواقع وأي معلومات أخرى ذات صلة
        
        مثال: "إنشاء عقد عمل بين شركة XYZ وجون دو لمنصب مستشار قانوني، براتب شهري قدره 1500 ريال عماني، ابتداءً من 1 يناير 2023، لمدة 2 سنوات."
        """
    
    # Document type selection
    doc_type = st.selectbox(doc_type_label, doc_types)
    
    # Initialize session state for voice input
    if "voice_input" not in st.session_state:
        st.session_state.voice_input = ""
    if "recording" not in st.session_state:
        st.session_state.recording = False
    
    # Text input for document specifications
    specs = st.text_area(spec_input_label, value=st.session_state.voice_input, 
                         placeholder=spec_placeholder, height=200)
    
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
    
    # Process document creation
    if st.button(create_button_text):
        if not specs:
            return
        
        with st.spinner(loading_text):
            # Generate document content
            document_content = llm_manager.generate_legal_document(doc_type, specs, language)
            
            # Create document title
            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            doc_title = f"{doc_type} - {current_date}"
            
            # Generate PDF
            pdf_buffer = pdf_gen.generate_legal_document(doc_title, document_content)
            
            # Generate certificate/document image
            image_buffer = img_gen.create_document_preview(doc_title, document_content[:500])
            
            # Display the document
            st.write("### Document Preview")
            st.write(document_content)
            
            # Download options
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label=pdf_download_text,
                    data=pdf_buffer,
                    file_name=f"{doc_type.replace(' ', '_').lower()}_{current_date}.pdf",
                    mime="application/pdf"
                )
            
            with col2:
                st.download_button(
                    label=image_download_text,
                    data=image_buffer,
                    file_name=f"{doc_type.replace(' ', '_').lower()}_{current_date}.png",
                    mime="image/png"
                )
            
            # Display image preview
            st.write("### Document Image Preview")
            st.image(image_buffer, width=400)
