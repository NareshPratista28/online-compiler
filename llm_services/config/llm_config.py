import os
from django.conf import settings

class LLMConfig:
    # Ollama Configuration
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "")
    OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.85"))
    
    # Gemini Configuration
    USE_GEMINI = os.getenv("USE_GEMINI", "true").lower() == "true"
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))

    
    # Timeout Configuration - Set to None to disable timeout
    OLLAMA_TIMEOUT = None  # No timeout
    OLLAMA_REQUEST_TIMEOUT = None  # No request timeout
    
    # Retry Configuration - Set to 0 to disable retries
    OLLAMA_MAX_RETRIES = 0  # No retries
    
    # Connection Configuration
    OLLAMA_DISABLE_TIMEOUTS = os.getenv("OLLAMA_DISABLE_TIMEOUTS", "true").lower() == "true"
    OLLAMA_DISABLE_RETRIES = os.getenv("OLLAMA_DISABLE_RETRIES", "true").lower() == "true"
    
    # File Paths
    COMPILER_BASE_PATH = os.path.join(settings.BASE_DIR, "java_files")
    TEST_CASES_PATH = os.path.join(COMPILER_BASE_PATH, "test_cases")
    MASTER_FILES_PATH = os.path.join(COMPILER_BASE_PATH, "master_files")
    
    # RAG Configuration
    VECTOR_STORE_PATH = os.path.join(settings.BASE_DIR, "chroma_db")
