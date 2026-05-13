"""
LLM Services Module
Contains all service classes for LLM operations
"""

# Hapus import langsung untuk menghindari error
# from .llm_service import LLMService

# Gunakan lazy import function instead
def get_llm_service():
    """Lazy import to avoid circular dependency issues"""
    from .llm_service import LLMService
    return LLMService()