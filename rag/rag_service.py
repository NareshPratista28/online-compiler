from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Any, Tuple
import logging
import os
from django.conf import settings

class RAGService:
    """
    RAG Service for retrieving relevant Java programming examples
    Migrated from FastAPI to Django
    Enhanced with retrieval evaluation capabilities
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.vector_store = None
        self._load_vector_store()
        
        # Note: Evaluation components removed to fix import dependency
        # Use template_integrated_evaluation.py for evaluation instead
    
    def _load_vector_store(self):
        """Load vector store from local ChromaDB"""
        try:
            # Get path to ChromaDB directory
            persist_directory = os.path.join(settings.BASE_DIR, "chroma_db")
            
            # Check if vector store exists
            if os.path.exists(persist_directory):
                self.vector_store = Chroma(
                    collection_name="java_examples",
                    embedding_function=self.embeddings,
                    persist_directory=persist_directory,
                )
                self.logger.info("[SUCCESS] Vector store loaded successfully")
            else:
                self.logger.warning(f"⚠️ Vector store not found at {persist_directory}")
                self.vector_store = None
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load vector store: {str(e)}")
            self.vector_store = None
    
    def get_relevant_examples(self, topic: str, additional_context: str = "", n: int = 2) -> List[Dict[str, Any]]:
        """
        Get relevant Java programming examples for a given topic
        
        Args:
            topic: Java programming topic to search for
            additional_context: Additional context to enrich the search
            n: Number of examples to retrieve
            
        Returns:
            List of relevant programming examples
        """
        if not self.vector_store:
            self.logger.warning("Vector store not available, cannot get relevant examples")
            return []
            
        try:
            # Create enriched query by combining topic and additional context
            query = topic
            
            if additional_context and len(additional_context.strip()) > 0:
                # Take max 200 characters from additional context to enrich query
                content = additional_context.strip()[:200]
                query = f"{topic} {content}"
                
            self.logger.info(f"Searching examples with ChromaDB: '{query[:50]}...'")

            # Search for relevant documents
            docs = self.vector_store.similarity_search(query, k=n)
            
            # Extract metadata and prepare examples for return
            examples = []
            for doc in docs:
                example = {
                    "studi_kasus": doc.metadata.get("studi_kasus", ""),
                    "tugas": doc.metadata.get("tugas", ""),
                    "code": doc.metadata.get("code", ""),
                    "topic": doc.metadata.get("topic", ""),
                }
                examples.append(example)
                
            self.logger.info(f"[SUCCESS] Successfully retrieved {len(examples)} relevant examples")
            return examples
            
        except Exception as e:
            self.logger.error(f"❌ Error while searching relevant examples: {str(e)}")
            return []
    
    def create_fallback_examples(self, topic: str) -> List[Dict[str, Any]]:
        """
        Create fallback examples when RAG is not available
        """
        fallback_examples = [
            {
                "studi_kasus": f"Implementasi dasar {topic} dalam Java",
                "tugas": f"- Buat implementasi {topic}\n- Pastikan kode dapat dikompilasi\n- Tambahkan dokumentasi yang sesuai",
                "code": f"public class {topic.replace(' ', '')} {{\n    // TODO: Implementasikan {topic}\n}}",
                "topic": topic
            }
        ]
        
        self.logger.info(f"Created fallback example for topic: {topic}")
        return fallback_examples
    
    def is_available(self) -> bool:
        """
        Check if RAG service is available
        """
        return self.vector_store is not None
    
    def get_relevant_examples_with_ids(self, topic: str, additional_context: str = "", n: int = 2) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Get relevant Java programming examples with document IDs for evaluation
        
        Args:
            topic: Java programming topic to search for
            additional_context: Additional context to enrich the search
            n: Number of examples to retrieve
            
        Returns:
            Tuple of (examples_list, document_ids_list)
        """
        if not self.vector_store:
            self.logger.warning("Vector store not available, cannot get relevant examples")
            return [], []
            
        try:
            # Create enriched query by combining topic and additional context
            query = topic
            
            if additional_context and len(additional_context.strip()) > 0:
                # Take max 200 characters from additional context to enrich query
                content = additional_context.strip()[:200]
                query = f"{topic} {content}"
                
            self.logger.info(f"Searching examples with ChromaDB: '{query[:50]}...'")

            # Search for relevant documents
            docs = self.vector_store.similarity_search(query, k=n)
            
            # Extract metadata, document IDs, and prepare examples
            examples = []
            doc_ids = []
            
            for i, doc in enumerate(docs):
                # Create document ID from metadata or index
                doc_id = doc.metadata.get("id", f"doc_{i}_{hash(doc.page_content) % 10000}")
                doc_ids.append(doc_id)
                
                example = {
                    "studi_kasus": doc.metadata.get("studi_kasus", ""),
                    "tugas": doc.metadata.get("tugas", ""),
                    "code": doc.metadata.get("code", ""),
                    "topic": doc.metadata.get("topic", ""),
                    "doc_id": doc_id  # Include document ID for evaluation
                }
                examples.append(example)
                
            self.logger.info(f"[SUCCESS] Successfully retrieved {len(examples)} examples with IDs: {doc_ids}")
            return examples, doc_ids
            
        except Exception as e:
            self.logger.error(f"❌ Error while searching relevant examples with IDs: {str(e)}")
            return [], []
    
    def get_vector_store_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store for evaluation context
        """
        if not self.vector_store:
            return {'error': 'Vector store not available'}
        
        try:
            # Get collection info
            collection = self.vector_store._collection
            all_docs = collection.get()
            
            doc_ids = all_docs.get('ids', [])
            metadatas = all_docs.get('metadatas', [])
            
            # Analyze topics if available
            topics = []
            for metadata in metadatas:
                if metadata and 'topic' in metadata:
                    topics.append(metadata['topic'])
            
            unique_topics = list(set(topics))
            
            stats = {
                'total_documents': len(doc_ids),
                'total_topics': len(unique_topics),
                'unique_topics': unique_topics[:10],  # Show first 10 topics
                'sample_doc_ids': doc_ids[:5],        # Show first 5 doc IDs
                'has_metadata': len([m for m in metadatas if m]) > 0,
                'collection_name': getattr(collection, 'name', 'unknown')
            }
            
            self.logger.info(f"📊 Vector store stats: {stats['total_documents']} docs, {stats['total_topics']} topics")
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Error getting vector store statistics: {str(e)}")
            return {'error': str(e)}
