import os
import faiss
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tensorflow_hub as hub
from langchain.schema import Document

class VectorStore:
    """Class to manage document embeddings and vector search"""
    
    def __init__(self, documents, use_huggingface=True):
        """
        Initialize vector store with documents.
        
        Args:
            documents: List of LangChain Document objects
            use_huggingface: Whether to use HuggingFace embeddings (True) or TensorFlow Hub (False)
        """
        self.documents = documents
        self.use_huggingface = use_huggingface
        
        # Initialize embeddings model
        if use_huggingface:
            # Use HuggingFace embeddings for multilingual support
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            # Create FAISS index from documents
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            # Use Universal Sentence Encoder (supports Arabic and English)
            self.embeddings = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
            self._create_faiss_index()
    
    def _create_faiss_index(self):
        """Create a FAISS index manually when using TensorFlow Hub embeddings"""
        # Extract text content from documents
        texts = [doc.page_content for doc in self.documents]
        
        # Generate embeddings
        embeddings = self.embeddings(texts).numpy()
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        self.index = index
        self.embeddings_array = embeddings
    
    def search(self, query, k=5):
        """
        Search for relevant documents.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        if self.use_huggingface:
            return self.vector_store.similarity_search_with_score(query, k=k)
        else:
            # Embed the query
            query_embedding = self.embeddings([query]).numpy()
            
            # Search the index
            distances, indices = self.index.search(query_embedding, k)
            
            # Format the results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= len(self.documents):  # Guard against out-of-bounds
                    continue
                doc = self.documents[idx]
                score = 1.0 / (1.0 + distances[0][i])  # Convert distance to similarity score
                results.append((doc, score))
            
            return results
    
    def search_by_law(self, query, law_name, k=5):
        """
        Search for relevant documents within a specific law.
        
        Args:
            query: Query string
            law_name: Name of the law to search within
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        # Filter documents by law name
        filtered_docs = [doc for doc in self.documents if doc.metadata.get("law_name") == law_name]
        
        if not filtered_docs:
            return []
        
        # Create a temporary vector store with filtered documents
        if self.use_huggingface:
            temp_vector_store = FAISS.from_documents(filtered_docs, self.embeddings)
            return temp_vector_store.similarity_search_with_score(query, k=k)
        else:
            # Extract text content from filtered documents
            texts = [doc.page_content for doc in filtered_docs]
            
            # Generate embeddings
            embeddings = self.embeddings(texts).numpy()
            
            # Create temporary FAISS index
            dimension = embeddings.shape[1]
            temp_index = faiss.IndexFlatL2(dimension)
            temp_index.add(embeddings)
            
            # Embed the query
            query_embedding = self.embeddings([query]).numpy()
            
            # Search the index
            distances, indices = temp_index.search(query_embedding, k)
            
            # Format the results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= len(filtered_docs):  # Guard against out-of-bounds
                    continue
                doc = filtered_docs[idx]
                score = 1.0 / (1.0 + distances[0][i])  # Convert distance to similarity score
                results.append((doc, score))
            
            return results
