import os
from openai import OpenAI

class OpenAIManager:
    """Manager for interactions with OpenAI models"""
    
    def __init__(self):
        """Initialize the OpenAI manager"""
        # Get API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Initialize the OpenAI client
        self.client = OpenAI(api_key=api_key)
        # The newest OpenAI model is "gpt-4o" which was released May 13, 2024
        # do not change this unless explicitly requested by the user
        self.model_name = "gpt-4o"
    
    def _generate_response(self, prompt, language="English"):
        """
        Generate a response from OpenAI.
        
        Args:
            prompt: The prompt to send to OpenAI
            language: The language of the error message
            
        Returns:
            The model's response as a string
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            error_message = f"Error generating content: {str(e)}"
            print(error_message)
            if language == "English":
                return "I apologize, but I encountered an error while generating a response. Please try again or contact support if the issue persists."
            else:
                return "أعتذر، لكنني واجهت خطأ أثناء إنشاء استجابة. يرجى المحاولة مرة أخرى أو الاتصال بالدعم إذا استمرت المشكلة."
    
    def answer_legal_question(self, question, context, language="English"):
        """
        Answer a legal question based on provided context.
        
        Args:
            question: The user's question
            context: Legal context information
            language: The language to respond in (English or Arabic)
            
        Returns:
            The model's response
        """
        if language == "English":
            prompt = f"""
            You are an expert legal assistant specializing in Omani law. Answer the following question based ONLY on the legal information provided. 
            If you cannot find a clear answer in the context, state that you cannot provide a definitive answer based on the available information.
            Do not invent or assume any legal information that is not in the context.
            
            LEGAL CONTEXT:
            {context}
            
            QUESTION:
            {question}
            
            ANSWER:
            """
        else:  # Arabic
            prompt = f"""
            أنت مساعد قانوني خبير متخصص في القانون العماني. أجب على السؤال التالي استنادًا فقط إلى المعلومات القانونية المقدمة.
            إذا لم تتمكن من العثور على إجابة واضحة في السياق، اذكر أنك لا تستطيع تقديم إجابة حاسمة بناءً على المعلومات المتاحة.
            لا تخترع أو تفترض أي معلومات قانونية غير موجودة في السياق.
            
            السياق القانوني:
            {context}
            
            السؤال:
            {question}
            
            الإجابة:
            """
        
        return self._generate_response(prompt, language)
    
    def summarize_article(self, article_text, language="English"):
        """
        Summarize a legal article in 3-5 lines.
        
        Args:
            article_text: The text of the legal article
            language: The language to respond in (English or Arabic)
            
        Returns:
            A concise summary of the article
        """
        if language == "English":
            prompt = f"""
            Summarize the following legal article in 3-5 concise lines. Focus on the main legal provisions and implications.
            
            ARTICLE:
            {article_text}
            
            SUMMARY:
            """
        else:  # Arabic
            prompt = f"""
            لخص المادة القانونية التالية في 3-5 أسطر موجزة. ركز على الأحكام والآثار القانونية الرئيسية.
            
            المادة:
            {article_text}
            
            الملخص:
            """
        
        return self._generate_response(prompt, language)
    
    def compare_laws(self, law1_name, law1_content, law2_name, law2_content, language="English"):
        """
        Compare two laws and provide analysis.
        
        Args:
            law1_name: Name of the first law
            law1_content: Content of the first law
            law2_name: Name of the second law
            law2_content: Content of the second law
            language: The language to respond in (English or Arabic)
            
        Returns:
            A comparison analysis of the two laws
        """
        if language == "English":
            prompt = f"""
            You are an expert in Omani law. Compare the following two laws, highlighting key similarities and differences in their provisions, scope, and legal implications.
            Structure your comparison with clear sections for similarities, differences, and a brief conclusion.
            
            FIRST LAW - {law1_name}:
            {law1_content}
            
            SECOND LAW - {law2_name}:
            {law2_content}
            
            COMPARISON:
            """
        else:  # Arabic
            prompt = f"""
            أنت خبير في القانون العماني. قارن بين القانونين التاليين، مع تسليط الضوء على أوجه التشابه والاختلاف الرئيسية في أحكامهما ونطاقهما وآثارهما القانونية.
            قم بهيكلة المقارنة بأقسام واضحة لأوجه التشابه والاختلاف وخاتمة موجزة.
            
            القانون الأول - {law1_name}:
            {law1_content}
            
            القانون الثاني - {law2_name}:
            {law2_content}
            
            المقارنة:
            """
        
        return self._generate_response(prompt, language)
    
    def generate_legal_document(self, document_type, specifications, language="English"):
        """
        Generate a legal document based on specifications.
        
        Args:
            document_type: Type of document to generate (e.g., "employment contract")
            specifications: User specifications for the document
            language: The language to respond in (English or Arabic)
            
        Returns:
            Generated legal document text
        """
        if language == "English":
            prompt = f"""
            As a legal expert in Omani law, create a professionally formatted {document_type} based on the following specifications.
            Ensure the document complies with Omani legal standards and includes all necessary clauses, provisions, and legal language.
            Format the document with proper sections, numbering, and structure.
            
            DOCUMENT TYPE: {document_type}
            
            SPECIFICATIONS:
            {specifications}
            
            DOCUMENT:
            """
        else:  # Arabic
            prompt = f"""
            بصفتك خبيرًا قانونيًا في القانون العماني، قم بإنشاء {document_type} منسق بشكل احترافي بناءً على المواصفات التالية.
            تأكد من أن المستند يتوافق مع المعايير القانونية العمانية ويتضمن جميع البنود والأحكام واللغة القانونية الضرورية.
            قم بتنسيق المستند بأقسام وترقيم وهيكل مناسب.
            
            نوع المستند: {document_type}
            
            المواصفات:
            {specifications}
            
            المستند:
            """
        
        return self._generate_response(prompt, language)
    
    def analyze_legal_case(self, case_description, legal_context, language="English"):
        """
        Analyze a legal case based on relevant laws.
        
        Args:
            case_description: Description of the legal case
            legal_context: Relevant legal context/laws
            language: The language to respond in (English or Arabic)
            
        Returns:
            Legal analysis of the case
        """
        if language == "English":
            prompt = f"""
            As an expert in Omani law, analyze the following legal case. Identify the relevant legal issues, apply the applicable laws from the provided context, and provide a legal assessment.
            Structure your analysis with these sections:
            1. Summary of Facts
            2. Legal Issues
            3. Applicable Laws
            4. Legal Analysis
            5. Conclusion
            
            CASE DESCRIPTION:
            {case_description}
            
            LEGAL CONTEXT:
            {legal_context}
            
            ANALYSIS:
            """
        else:  # Arabic
            prompt = f"""
            بصفتك خبيرًا في القانون العماني، قم بتحليل القضية القانونية التالية. حدد القضايا القانونية ذات الصلة، وطبق القوانين المعمول بها من السياق المقدم، وقدم تقييمًا قانونيًا.
            قم بهيكلة تحليلك بهذه الأقسام:
            1. ملخص الوقائع
            2. القضايا القانونية
            3. القوانين المطبقة
            4. التحليل القانوني
            5. الخلاصة
            
            وصف القضية:
            {case_description}
            
            السياق القانوني:
            {legal_context}
            
            التحليل:
            """
        
        return self._generate_response(prompt, language)