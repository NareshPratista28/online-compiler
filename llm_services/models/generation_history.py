import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from django.db import connection

class GenerationHistoryModel:
    """Model untuk menyimpan history generasi soal - Migrated from FastAPI"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def save_generation(self, content_id: int, topic_title: str, result: dict, generation_time: float, 
                       user_email: str = None, question_id: int = None, score: int = 0) -> int:
        """
        Menyimpan hasil generasi soal ke dalam database.
        
        Args:
            content_id: ID konten pembelajaran
            topic_title: Judul/topik materi
            result: Dictionary hasil generasi (raw_response, code, studi_kasus, tugas)
            generation_time: Waktu pemrosesan dalam detik
            
        Returns:
            ID history yang tersimpan atau 0 jika gagal
        """
        try:
            with connection.cursor() as cursor:
                # Tambahkan metadata tambahan ke result JSON karena kolom database belum tersedia
                if isinstance(result, dict):
                    if question_id: result['question_id'] = question_id
                    if user_email: result['user_email'] = user_email
                    if score: result['score'] = score
                
                # Update result_json
                result_json = json.dumps(result, ensure_ascii=False)
                created_at = datetime.now()

                # Simpan ke database (Hanya kolom yang pasti ada)
                query = """
                    INSERT INTO generation_history 
                    (content_id, topic_title, result, generation_time, created_at) 
                    VALUES (%s, %s, %s, %s, %s)
                """
                
                cursor.execute(query, (content_id, topic_title, result_json, generation_time, created_at))
                
                # Ambil ID yang baru saja dimasukkan
                history_id = cursor.lastrowid
                self.logger.info(f"Generation history saved with ID: {history_id}")
                return history_id
                
        except Exception as e:
            self.logger.error(f"Error saving generation history: {str(e)}")
            return 0
    
    def get_history_by_id(self, history_id: int) -> Optional[Dict[str, Any]]:
        """
        Mengambil detail history berdasarkan ID.
        
        Args:
            history_id: ID history
            
        Returns:
            Dictionary berisi detail history atau None jika tidak ditemukan
        """
        try:
            with connection.cursor() as cursor:
                query = """
                    SELECT h.id, h.content_id, c.title as content_title, 
                           h.topic_title, h.result, h.generation_time, h.created_at
                    FROM generation_history h
                    JOIN contents c ON h.content_id = c.id
                    WHERE h.id = %s
                """
                
                cursor.execute(query, [history_id])
                result = cursor.fetchone()
                
                if result:
                    history_data = {
                        'id': result[0],
                        'content_id': result[1],
                        'content_title': result[2],
                        'topic_title': result[3],
                        'result': json.loads(result[4]) if result[4] else {},
                        'generation_time': result[5],
                        'created_at': result[6]
                    }
                    return history_data
                else:
                    self.logger.warning(f"History not found for ID: {history_id}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error fetching history by ID: {str(e)}")
            return None
    
    def get_history_list(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Mengambil daftar history generasi.
        
        Args:
            limit: Jumlah maksimum record yang diambil
            offset: Offset untuk pagination
            
        Returns:
            List dictionary berisi history
        """
        try:
            with connection.cursor() as cursor:
                query = """
                    SELECT h.id, h.content_id, c.title as content_title, 
                           h.topic_title, h.generation_time, h.created_at, h.result
                    FROM generation_history h
                    JOIN contents c ON h.content_id = c.id
                    ORDER BY h.created_at DESC
                    LIMIT %s OFFSET %s
                """
                
                cursor.execute(query, [limit, offset])
                results = cursor.fetchall()
                
                history_list = []
                for result in results:
                    # Parse result JSON to check for JUnit test
                    has_junit_test = False
                    try:
                        result_data = json.loads(result[6]) if result[6] else {}
                        has_junit_test = bool(result_data.get('junit_test_code'))
                    except (json.JSONDecodeError, TypeError):
                        has_junit_test = False
                    
                    history_list.append({
                        'id': result[0],
                        'content_id': result[1],
                        'content_title': result[2],
                        'topic_title': result[3],
                        'generation_time': result[4],
                        'created_at': result[5],
                        'has_junit_test': has_junit_test
                    })
                
                return history_list
                
        except Exception as e:
            self.logger.error(f"Error fetching history list: {str(e)}")
            return []
    
    def get_history_by_content_id(self, content_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Mengambil daftar history generasi untuk konten tertentu.
        
        Args:
            content_id: ID konten pembelajaran
            limit: Jumlah maksimum record yang diambil
            
        Returns:
            List dictionary berisi history
        """
        try:
            with connection.cursor() as cursor:
                query = """
                    SELECT h.id, h.content_id, c.title as content_title,
                           h.topic_title, h.generation_time, h.created_at, h.result
                    FROM generation_history h
                    JOIN contents c ON h.content_id = c.id
                    WHERE h.content_id = %s
                    ORDER BY h.created_at DESC
                    LIMIT %s
                """
                
                cursor.execute(query, [content_id, limit])
                results = cursor.fetchall()
                
                history_list = []
                for result in results:
                    # Parse result JSON to check for JUnit test
                    has_junit_test = False
                    try:
                        result_data = json.loads(result[6]) if result[6] else {}
                        has_junit_test = bool(result_data.get('junit_test_code'))
                    except (json.JSONDecodeError, TypeError):
                        has_junit_test = False
                        
                    history_list.append({
                        'id': result[0],
                        'content_id': result[1],
                        'content_title': result[2],
                        'topic_title': result[3],
                        'generation_time': result[4],
                        'created_at': result[5],
                        'has_junit_test': has_junit_test
                    })
                
                return history_list
                
        except Exception as e:
            self.logger.error(f"Error fetching history by content ID: {str(e)}")
            return []
    
    def search_history(self, search_term: str, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Mencari history generasi berdasarkan judul topik.
        
        Args:
            search_term: Kata kunci pencarian untuk judul topik
            limit: Jumlah maksimum record yang diambil
            offset: Offset untuk pagination
            
        Returns:
            List dictionary berisi history yang sesuai
        """
        try:
            with connection.cursor() as cursor:
                query = """
                    SELECT h.id, h.content_id, c.title as content_title, 
                           h.topic_title, h.generation_time, h.created_at, h.result
                    FROM generation_history h
                    JOIN contents c ON h.content_id = c.id
                    WHERE h.topic_title LIKE %s
                    ORDER BY h.created_at DESC
                    LIMIT %s OFFSET %s
                """
                
                search_pattern = f"%{search_term}%"
                cursor.execute(query, [search_pattern, limit, offset])
                results = cursor.fetchall()
                
                history_list = []
                for result in results:
                    # Parse result JSON to check for JUnit test
                    has_junit_test = False
                    try:
                        result_data = json.loads(result[6]) if result[6] else {}
                        has_junit_test = bool(result_data.get('junit_test_code'))
                    except (json.JSONDecodeError, TypeError):
                        has_junit_test = False
                        
                    history_list.append({
                        'id': result[0],
                        'content_id': result[1],
                        'content_title': result[2],
                        'topic_title': result[3],
                        'generation_time': result[4],
                        'created_at': result[5],
                        'has_junit_test': has_junit_test
                    })
                
                return history_list
                
        except Exception as e:
            self.logger.error(f"Error searching history: {str(e)}")
            return []
    
    def get_search_total_count(self, search_term: str) -> int:
        """
        Mendapatkan total jumlah hasil pencarian history berdasarkan judul topik.
        
        Args:
            search_term: Kata kunci pencarian untuk judul topik
            
        Returns:
            Total jumlah record history yang sesuai
        """
        try:
            with connection.cursor() as cursor:
                query = """
                    SELECT COUNT(*)
                    FROM generation_history h
                    JOIN contents c ON h.content_id = c.id
                    WHERE h.topic_title LIKE %s
                """
                search_pattern = f"%{search_term}%"
                cursor.execute(query, [search_pattern])
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            self.logger.error(f"Error getting search total count: {str(e)}")
            return 0

    def get_total_count(self) -> int:
        """
        Mendapatkan total jumlah history untuk pagination
        
        Returns:
            Total jumlah record history
        """
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM generation_history")
                result = cursor.fetchone()
                return result[0] if result else 0
                
        except Exception as e:
            self.logger.error(f"Error getting total count: {str(e)}")
            return 0
    
    def delete_history(self, history_id: int) -> bool:
        """
        Menghapus history berdasarkan ID
        
        Args:
            history_id: ID history yang akan dihapus
            
        Returns:
            True jika berhasil, False jika gagal
        """
        try:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM generation_history WHERE id = %s", [history_id])
                affected_rows = cursor.rowcount
                
                if affected_rows > 0:
                    self.logger.info(f"History with ID {history_id} deleted successfully")
                    return True
                else:
                    self.logger.warning(f"No history found with ID {history_id}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error deleting history: {str(e)}")
            return False
    
    def save_execution_result(self, user_email: str, question_id: int, student_code: str, 
                             test_result: dict, score: int = 0) -> int:
        """
        Menyimpan hasil eksekusi kode student ke tabel generation_history.
        
        Args:
            user_email: Email user yang mengirim kode
            question_id: ID pertanyaan yang dikerjakan
            student_code: Kode yang dikirim student
            test_result: Dictionary hasil test (output, error, passed, etc.)
            score: Score yang didapat student
            
        Returns:
            ID history yang tersimpan atau 0 jika gagal
        """
        try:
            with connection.cursor() as cursor:
                # Konversi hasil ke JSON untuk penyimpanan
                execution_data = {
                    'type': 'execution',
                    'student_code': student_code,
                    'test_result': test_result,
                    'user_email': user_email,
                    'question_id': question_id,
                    'score': score
                }
                result_json = json.dumps(execution_data, ensure_ascii=False)
                created_at = datetime.now()
                
                # Simpan ke database dengan topic_title yang menunjukkan ini adalah execution
                # Format topic_title penting untuk filtering nanti karena kolom user_email tidak ada
                topic_title = f"Code Execution - Question {question_id} - {user_email}"
                
                query = """
                    INSERT INTO generation_history 
                    (content_id, topic_title, result, generation_time, created_at) 
                    VALUES (%s, %s, %s, %s, %s)
                """
                
                # content_id = question_id untuk execution history
                cursor.execute(query, (question_id, topic_title, result_json, 0.0, created_at))
                
                # Ambil ID yang baru saja dimasukkan
                history_id = cursor.lastrowid
                self.logger.info(f"Execution history saved with ID: {history_id}")
                return history_id
                
        except Exception as e:
            self.logger.error(f"Error saving execution history: {str(e)}")
            return 0

    def update_execution_score(self, history_id: int, new_score: int) -> bool:
        """
        Update score untuk execution history tertentu.
        
        Args:
            history_id: ID history
            new_score: Score baru
            
        Returns:
            True jika berhasil, False jika gagal
        """
        try:
            with connection.cursor() as cursor:
                # 1. Ambil result saat ini karena kolom score tidak ada
                cursor.execute("SELECT result FROM generation_history WHERE id = %s", [history_id])
                row = cursor.fetchone()
                
                if not row:
                    self.logger.warning(f"No history found with ID {history_id}")
                    return False
                
                # 2. Update score di dalam JSON result
                result_json_str = row[0]
                try:
                    result_data = json.loads(result_json_str) if result_json_str else {}
                except json.JSONDecodeError:
                    result_data = {}
                    
                result_data['score'] = new_score
                new_result_json = json.dumps(result_data, ensure_ascii=False)
                
                # 3. Simpan kembali ke database
                cursor.execute(
                    "UPDATE generation_history SET result = %s WHERE id = %s", 
                    [new_result_json, history_id]
                )
                
                if cursor.rowcount > 0:
                    self.logger.info(f"Score updated for history ID {history_id}: {new_score}")
                    return True
                return False
                    
        except Exception as e:
            self.logger.error(f"Error updating execution score: {str(e)}")
            return False

    def get_execution_history_by_user(self, user_email: str, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Mengambil daftar execution history untuk user tertentu.
        
        Args:
            user_email: Email user
            limit: Jumlah maksimum record yang diambil
            offset: Offset untuk pagination
            
        Returns:
            List dictionary berisi execution history
        """
        try:
            with connection.cursor() as cursor:
                # Filter by topic_title karena kolom user_email tidak ada
                # topic_title format: "Code Execution - Question {question_id} - {user_email}"
                query = """
                    SELECT id, content_id, result, created_at
                    FROM generation_history
                    WHERE topic_title LIKE %s AND result LIKE '%"type": "execution"%'
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """
                
                email_pattern = f"%{user_email}%"
                cursor.execute(query, [email_pattern, limit, offset])
                results = cursor.fetchall()
                
                execution_list = []
                for result in results:
                    try:
                        result_data = json.loads(result[2]) if result[2] else {}
                        
                        # Ambil score dari internal JSON jika tidak ada di DB
                        score = result_data.get('score', 0)
                        
                        execution_list.append({
                            'id': result[0],
                            'question_id': result[1], # content_id holds question_id
                            'test_result': result_data.get('test_result', {}),
                            'score': score,
                            'created_at': result[3]
                        })
                    except (json.JSONDecodeError, TypeError):
                        continue
                
                return execution_list
                
        except Exception as e:
            self.logger.error(f"Error fetching execution history by user: {str(e)}")
            return []