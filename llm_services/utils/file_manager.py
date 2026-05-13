import os
import shutil
import logging
from typing import Optional, Dict, Any
from django.conf import settings

class FileManager:
    """
    File management utilities for LLM-generated Java files and JUnit tests
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    @staticmethod
    def ensure_directory_exists(directory_path: str) -> bool:
        """Ensure directory exists, create if it doesn't"""
        try:
            os.makedirs(directory_path, exist_ok=True)
            return True
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to create directory {directory_path}: {e}")
            return False
    
    def save_junit_file(self, junit_code: str, filename: str, target_dir: str) -> str:
        """Save JUnit test file to specified directory"""
        try:
            self.ensure_directory_exists(target_dir)
            file_path = os.path.join(target_dir, filename)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(junit_code)
                
            self.logger.info(f"✅ JUnit file saved: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save JUnit file {filename}: {e}")
            raise
    
    def save_java_template_file(self, code: str, filename: str, target_dir: str) -> str:
        """Save Java template file to specified directory"""
        try:
            self.ensure_directory_exists(target_dir)
            file_path = os.path.join(target_dir, filename)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code)
                
            self.logger.info(f"✅ Java template file saved: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save Java template file {filename}: {e}")
            raise
    
    def rename_junit_file(self, old_path: str, new_filename: str) -> str:
        """Rename JUnit file after getting question_id"""
        try:
            directory = os.path.dirname(old_path)
            new_path = os.path.join(directory, new_filename)
            
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
                self.logger.info(f"✅ File renamed: {old_path} → {new_path}")
                return new_path
            else:
                self.logger.warning(f"⚠️ File not found for rename: {old_path}")
                return old_path
                
        except Exception as e:
            self.logger.error(f"❌ Failed to rename file {old_path}: {e}")
            raise
    
    def copy_junit_to_test_cases(self, junit_code: str, class_name: str, 
                               question_id: Optional[int] = None, 
                               content_id: Optional[int] = None) -> str:
        """
        Copy JUnit test code to test_cases directory with proper naming
        """
        try:
            test_cases_dir = os.path.join(settings.BASE_DIR, "java_files", "test_cases")
            self.ensure_directory_exists(test_cases_dir)
            
            # Determine filename based on available IDs
            if question_id:
                filename = f"JUnit{class_name}Test_Q{question_id}.java"
            elif content_id:
                filename = f"JUnit{class_name}Test_C{content_id}.java"
            else:
                filename = f"JUnit{class_name}Test.java"
            
            file_path = self.save_junit_file(junit_code, filename, test_cases_dir)
            
            self.logger.info(f"✅ JUnit test copied to test_cases: {filename}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to copy JUnit to test_cases: {e}")
            raise
    
    def create_master_files(self, result: Dict[str, Any], class_name: str, content_id: int) -> Dict[str, str]:
        """
        Create master files for a generated question
        """
        try:
            master_dir = os.path.join(settings.BASE_DIR, "java_files", "master_files", str(content_id))
            self.ensure_directory_exists(master_dir)
            
            file_paths = {}
            
            # Create Java template file
            java_filename = f"{class_name}.java"
            java_path = self.save_java_template_file(
                result.get('code', ''), java_filename, master_dir
            )
            file_paths['java_template'] = java_path
            
            # Create JUnit test file
            junit_filename = f"JUnit{class_name}Test.java"
            junit_path = self.save_junit_file(
                result.get('junit_test_code', ''), junit_filename, master_dir
            )
            file_paths['junit_test'] = junit_path
            
            # Create solution template
            if result.get('solution_template'):
                solution_filename = f"{class_name}Solution.java"
                solution_path = self.save_java_template_file(
                    result.get('solution_template', ''), solution_filename, master_dir
                )
                file_paths['solution_template'] = solution_path
            
            # Create metadata file
            metadata_path = self.create_metadata_file(master_dir, class_name, content_id, result)
            file_paths['metadata'] = metadata_path
            
            self.logger.info(f"✅ Master files created for content_id {content_id}")
            return file_paths
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create master files: {e}")
            raise
    
    def create_metadata_file(self, master_dir: str, class_name: str, content_id: int, result: Dict[str, Any]) -> str:
        """Create metadata file for master files"""
        try:
            import json
            from datetime import datetime
            
            metadata = {
                "content_id": content_id,
                "class_name": class_name,
                "generated_at": datetime.now().isoformat(),
                "studi_kasus": result.get('studi_kasus', ''),
                "tugas": result.get('tugas', ''),
                "test_cases_count": len(result.get('test_cases', [])),
                "files_created": {
                    "java_template": f"{class_name}.java",
                    "junit_test": f"JUnit{class_name}Test.java",
                    "solution_template": f"{class_name}Solution.java"
                }
            }
            
            metadata_path = os.path.join(master_dir, "metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ Metadata file created: {metadata_path}")
            return metadata_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create metadata file: {e}")
            raise