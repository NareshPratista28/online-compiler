"""
LLM Services Models Package
Django models compatible with existing Laravel database structure

Note: Models are imported on-demand to avoid AppRegistryNotReady errors
"""

# Models will be imported individually as needed
# from .content import ContentModel
# from .generation_history import GenerationHistory

__all__ = ['ContentModel', 'GenerationHistory']
