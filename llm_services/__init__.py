"""
LLM Services Package
Handles all LLM-related operations for Django Online Compiler
"""

# Import dikurangi untuk menghindari circular imports saat startup
# Services akan di-import on-demand saat dibutuhkan

default_app_config = 'llm_services.apps.LlmServicesConfig'

# Lazy imports untuk menghindari error saat Django startup
def get_llm_service():
    """Lazy import LLMService to avoid startup errors"""
    from .services.llm_service import LLMService
    return LLMService