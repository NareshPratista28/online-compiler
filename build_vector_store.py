"""
Script untuk membuat vector store dari contoh pertanyaan Java.
"""

import sys
import os
import django
import logging

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'onlinecompiler.settings')

# Tambahkan project root ke Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Setup Django
django.setup()

# Import setelah Django setup
from rag.indexer import create_vector_store

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

if __name__ == "__main__":
    print("=== Memulai proses pembuatan vector store ===")
    success = create_vector_store()
    
    if success:
        print("=== Vector store berhasil dibuat! ===")
    else:
        print("=== Gagal membuat vector store! ===")
