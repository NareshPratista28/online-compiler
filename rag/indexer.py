from django.conf import settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import logging
import os

from data.java_examples import JAVA_EXAMPLES
from data.java_examples_expanded import EXPANDED_JAVA_EXAMPLES

ALL_EXAMPLES = JAVA_EXAMPLES + EXPANDED_JAVA_EXAMPLES

logger = logging.getLogger(__name__)

def create_vector_store():
    """Membuat vector store"""
    try:
        # Inisialisasi embeddings
        logger.info("Inisialisasi embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Gunakan VECTOR_STORE_PATH dari Django settings
        persist_directory = settings.VECTOR_STORE_PATH
        
        # Pastikan direktori untuk ChromaDB ada
        os.makedirs(persist_directory, exist_ok=True)
        logger.info(f"ChromaDB akan disimpan di: {persist_directory}")
        
        # Persiapkan documents untuk vectorstore
        logger.info("Menyiapkan dokumen...")
        documents = []
        metadatas = []
        ids = []
        
        for i, example in enumerate (ALL_EXAMPLES):
            # Create document content by combining all fields
            content = f"Topic: {example['topic']}\n\n"
            content += f"Studi Kasus: {example['studi_kasus']}\n\n"
            content += f"Tugas: {example['tugas']}\n\n"
            
            documents.append(content)
            metadatas.append({
                "topic": example["topic"],
                "studi_kasus": example["studi_kasus"],
                "tugas": example["tugas"],
                "code": example["code"]
            })
            ids.append(f"java_example_{i}")
        
        # Buat dan simpan vector store
        logger.info("Membuat vector store...")
        vector_store = Chroma.from_texts(
            texts=documents,
            embedding=embeddings,
            metadatas=metadatas,
            ids=ids,
            collection_name="java_examples",
            persist_directory=persist_directory
        )
        
        logger.info(f"Vector store berhasil dibuat dengan {len(documents)} dokumen dan disimpan di {persist_directory}")
        return True
        
    except Exception as e:
        logger.error(f"Error saat membuat vector store: {str(e)}")
        return False
