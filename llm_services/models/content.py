import logging
from typing import Dict, Any, Optional
from django.db import connection

class ContentModel:
    """Model untuk mengakses data content dari database - Migrated from FastAPI"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_prompt_and_description(self, content_id: int) -> Optional[Dict[str, Any]]:
        """
        Mengambil prompt_llm dan materi pembelajaran dari tabel content berdasarkan content_id
        
        Args:
            content_id: ID konten pembelajaran
            
        Returns:
            Dictionary berisi prompt_llm, description, dan title atau None jika tidak ditemukan
        """
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT id, title, description, prompt_llm, created_at, updated_at
                    FROM contents 
                    WHERE id = %s
                """, [content_id])
                result = cursor.fetchone()
                
                if result:
                    return {
                        'id': result[0],
                        'title': result[1],
                        'description': result[2],
                        'prompt_llm': result[3],
                        'created_at': result[4],
                        'updated_at': result[5]
                    }
                else:
                    self.logger.warning(f"Content not found for ID: {content_id}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error fetching content: {str(e)}")
            return None
    
    def get_by_id(self, content_id: int) -> Optional[Dict[str, Any]]:
        """
        Alias untuk get_prompt_and_description untuk konsistensi
        """
        return self.get_prompt_and_description(content_id)
    
    def get_all_contents(self, limit: int = 50, offset: int = 0) -> list:
        """
        Mengambil semua content dengan pagination
        
        Args:
            limit: Jumlah maksimal data yang diambil
            offset: Offset untuk pagination
            
        Returns:
            List dictionary berisi data content
        """
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT id, title, description, prompt_llm, created_at, updated_at
                    FROM contents 
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """, [limit, offset])
                
                rows = cursor.fetchall()
                
                contents = []
                for row in rows:
                    contents.append({
                        'id': row[0],
                        'title': row[1],
                        'description': row[2],
                        'prompt_llm': row[3],
                        'created_at': row[4],
                        'updated_at': row[5]
                    })
                
                self.logger.info(f"✅ Mengambil {len(contents)} content dari database")
                return contents
                
        except Exception as e:
            self.logger.error(f"❌ Error mengambil content list: {str(e)}")
            return []