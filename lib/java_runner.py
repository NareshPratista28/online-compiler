import subprocess
import os
import re
import logging
import json
import shutil
import time
from typing import Any, Dict, Optional
from lib.code_analyzer import CodeAnalyzer, ValidationResult
from pathlib import Path
from dataclasses import dataclass

@dataclass
class JavaRunnerConfig:
    main_directory: str = "java_files/"
    online_compiler_dir: str = os.getenv("ONLINE_COMPILER_DIR", ".")
    master_files_dir: str = "question_masters"
    test_cases_dir: str = "java_files/test_cases"

    def __post_init__(self):
        # Validate paths exist
        if not os.path.exists(self.online_compiler_dir):
            raise ValueError(f"Online compiler directory not found: {self.online_compiler_dir}")
        
        # Ensure test_cases directory exists
        os.makedirs(self.test_cases_dir, exist_ok=True)
            
    @property
    def master_files_path(self) -> Path:
        return Path(self.online_compiler_dir) / self.master_files_dir

class JavaRunnerError(Exception):
    """Base exception for JavaRunner operations"""
    pass

class FileSetupError(JavaRunnerError):
    """Error dalam file setup operations"""
    pass

class JavaRunner:
    def __init__(self, user_directory: str, code: str, question_data: Optional[Dict[str, Any]] = None, 
                 config: Optional[JavaRunnerConfig] = None):
        try:
            self.config = config or JavaRunnerConfig()
            self.user_directory = self.get_user_dir(user_directory)
            self.code = code
            self.question_data = question_data

            self._metadata_cache = {}  # Cache for question metadata

            # Setup logging with UTF-8 encoding for Windows compatibility
            self.logger = logging.getLogger(f"{__name__}.{self.user_directory}")
            self.logger.setLevel(logging.INFO)
            
            # Create handler with UTF-8 encoding to handle emoji characters
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setLevel(logging.INFO)
                formatter = logging.Formatter('%(levelname)s %(message)s')
                handler.setFormatter(formatter)
                
                # Try to set UTF-8 encoding for Windows compatibility
                try:
                    if hasattr(handler.stream, 'reconfigure'):
                        handler.stream.reconfigure(encoding='utf-8')
                except:
                    pass
                    
                self.logger.addHandler(handler)
            
            # Initialize filename from code if possible
            self.filename = self._extract_class_name_from_code(code)
            self.test_filename = f"JUnit{self.filename}Test" if self.filename else None
            
            # Initialize package_name - default to user_directory if not set elsewhere
            self.package_name = self.user_directory
            
            # Extract question_id if available
            self.question_id = None
            if question_data and isinstance(question_data, dict):
                self.question_id = question_data.get('question_id')
                
            print("[OK] JavaRunner initialized successfully for user: {user_directory}")
            
        except Exception as e:
            print(f"[ERROR] Error initializing JavaRunner: {e}")
            raise
            
    def get_user_dir(self, user_mail):
        replace_at = re.sub('@+', "_", user_mail)
        replace_dot = re.sub('\.', "_", replace_at)
        return replace_dot
        
    def check_and_create_dir(self):
        directory_name = f"{self.config.main_directory}/{self.user_directory}"        
        if not os.path.isdir(directory_name):
            os.makedirs(directory_name, exist_ok=True)
            self.logger.info(f"Created user directory: {directory_name}")

    def load_junit_test_from_file(self, question_id: int) -> Optional[Dict[str, Any]]:
        """
        Load JUnit test file based on question_id from saved files
        This is the NEW implementation for the LLM system
        """
        try:
            test_cases_dir = self.config.test_cases_dir
            
            # Ensure directory exists
            if not os.path.exists(test_cases_dir):
                self.logger.warning(f"Test cases directory not found: {test_cases_dir}")
                return None
            
            import glob
            
            # Pattern 1: Primary pattern - JUnit{ClassName}Test_Q{question_id}.java
            pattern1 = f"{test_cases_dir}/JUnit*Test_Q{question_id}.java"
            matching_files = glob.glob(pattern1)
            
            # Pattern 2: Alternative pattern - JUnit{ClassName}Test_{question_id}.java  
            if not matching_files:
                pattern2 = f"{test_cases_dir}/JUnit*Test_{question_id}.java"
                matching_files = glob.glob(pattern2)
            
            # Pattern 3: Broader search for debugging
            if not matching_files:
                pattern3 = f"{test_cases_dir}/*{question_id}*.java"
                matching_files = glob.glob(pattern3)
            
            if matching_files:
                junit_filename = matching_files[0]
                self.logger.info(f"✅ Found JUnit file: {os.path.basename(junit_filename)}")
                
                with open(junit_filename, 'r', encoding='utf-8') as f:
                    junit_code = f.read()
                
                # Parse JUnit code and return structured data
                test_data = self.parse_junit_code(junit_code)
                test_data['junit_file_path'] = junit_filename
                test_data['question_id'] = question_id
                
                self.logger.info(f"✅ Successfully loaded JUnit test for question_id: {question_id}")
                return test_data
            else:
                self.logger.warning(f"⚠️ No JUnit file found for question_id: {question_id}")
                self._debug_test_files_directory(test_cases_dir, question_id)
                return None
                    
        except Exception as e:
            self.logger.error(f"❌ Error loading JUnit test from file: {e}")
            return None

    def _debug_test_files_directory(self, test_cases_dir: str, question_id: int):
        """Debug helper to show available test files"""
        try:
            self.logger.info(f"📁 Debugging test files directory: {test_cases_dir}")
            self.logger.info(f"🔍 Looking for question_id: {question_id}")
            
            if os.path.exists(test_cases_dir):
                all_files = os.listdir(test_cases_dir)
                junit_files = [f for f in all_files if f.endswith('.java') and 'JUnit' in f]
                
                self.logger.info(f"📋 Available JUnit test files ({len(junit_files)}):")
                for file in junit_files:
                    self.logger.info(f"   - {file}")
                    
                if not junit_files:
                    self.logger.warning("❌ No JUnit test files found in directory")
            else:
                self.logger.error(f"❌ Test cases directory does not exist: {test_cases_dir}")
                
        except Exception as e:
            self.logger.error(f"❌ Error debugging test files directory: {e}")

    def load_default_junit_test(self) -> Optional[Dict[str, Any]]:
        """
        Load default JUnit test from JUnitBankTest.java.txt as fallback
        """
        try:
            default_file = os.path.join(self.config.test_cases_dir, "JUnitBankTest.java.txt")
            if os.path.exists(default_file):
                self.logger.info("📄 Loading default JUnit test file")
                with open(default_file, 'r', encoding='utf-8') as f:
                    junit_code = f.read()
                return self.parse_junit_code(junit_code)
            else:
                self.logger.warning(f"⚠️ Default JUnit test file not found: {default_file}")
                return None
        except Exception as e:
            self.logger.error(f"❌ Error loading default JUnit test: {e}")
            return None

    def parse_junit_code(self, junit_code: str) -> Dict[str, Any]:
        """
        Parse JUnit code to extract test information
        Enhanced version for better compatibility
        """
        try:
            # Extract class name
            class_match = re.search(r'public class (\w+)', junit_code)
            class_name = class_match.group(1) if class_match else "TestClass"
            
            # Extract test methods
            test_methods = re.findall(r'@Test[^}]*public void (\w+)\([^}]*\{[^}]*\}', junit_code, re.DOTALL)
            
            # Extract package if any
            package_match = re.search(r'package ([^;]+);', junit_code)
            package_name = package_match.group(1) if package_match else ""
            
            # Extract target class being tested
            target_class_match = re.search(r'(\w+)\.main\(|new (\w+)\(', junit_code)
            target_class = None
            if target_class_match:
                target_class = target_class_match.group(1) or target_class_match.group(2)
            
            return {
                'junit_test_code': junit_code,
                'class_name': class_name,
                'package_name': package_name,
                'target_class': target_class,
                'test_methods': test_methods,
                'test_cases': [{'method': method, 'description': f'Test method: {method}'} for method in test_methods]
            }
        except Exception as e:
            self.logger.error(f"❌ Error parsing JUnit code: {e}")
            return {
                'junit_test_code': junit_code,
                'test_cases': []
            }

    def setup_junit_test_file(self, question_id: int) -> bool:
        """
        Setup JUnit test file from database or file system
        
        Args:
            question_id: Question ID to load test for
            
        Returns:
            bool: True if JUnit test was successfully set up
        """
        try:
            # First, check if we have JUnit code in question_data (from database)
            if (self.question_data and 
                isinstance(self.question_data, dict) and 
                self.question_data.get('use_database_junit') and
                self.question_data.get('junit_tests')):
                
                self.logger.info(f"📚 Setting up JUnit test from database for question_id: {question_id}")
                return self._create_junit_test_from_database()
            
            # Fallback to file-based JUnit loading
            self.logger.info(f"📁 Attempting to load JUnit test from file for question_id: {question_id}")
            test_data = self.load_junit_test_from_file(question_id)
            
            if test_data:
                return self._create_junit_test_from_data(test_data)
            else:
                self.logger.warning(f"⚠️ No JUnit test found for question_id: {question_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error setting up JUnit test: {e}")
            return False

    def _create_junit_test_from_database(self) -> bool:
        """
        Create JUnit test file from database JUnit code
        
        Returns:
            bool: True if test file was created successfully
        """
        try:
            junit_code = self.question_data.get('junit_tests', '')
            class_name = self.question_data.get('class_name', self.filename)
            
            if not junit_code:
                self.logger.error("❌ No JUnit code provided in question_data")
                return False
            
            self.logger.info(f"Creating JUnit test from database for class: {class_name}")
            
            # Process JUnit code and get actual class name
            processed_junit_code, actual_class_name = self._process_junit_code_for_package(junit_code, class_name)
            
            # Use actual class name from code for filename
            self.test_filename = actual_class_name
            
            # Create JUnit test file in user directory
            test_file_path = f"{self.config.main_directory}/{self.user_directory}/{self.test_filename}.java"
            
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write(processed_junit_code)
            
            # Store test file info
            self.test_file_path = test_file_path
            self.test_class_name = actual_class_name
            
            self.logger.info(f"JUnit test file created from database: {test_file_path}")
            self.logger.info(f"Test class name: {actual_class_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating JUnit test from database: {e}")
            return False

    def _create_junit_test_from_data(self, test_data: Dict[str, Any]) -> bool:
        """
        Create JUnit test file from parsed test data
        
        Args:
            test_data: Parsed JUnit test data
            
        Returns:
            bool: True if test file was created successfully
        """
        try:
            junit_code = test_data.get('junit_test_code', '')
            class_name = test_data.get('class_name', 'TestClass')
            
            if not junit_code:
                self.logger.error("❌ No JUnit code in test data")
                return False
            
            self.logger.info(f"🧪 Creating JUnit test from data for class: {class_name}")
            
            # Process JUnit code and get actual class name
            processed_junit_code, actual_class_name = self._process_junit_code_for_package(junit_code, class_name)
            
            # Use actual class name from code for filename
            self.test_filename = actual_class_name
            
            # Create test file path
            test_file_path = f"{self.config.main_directory}/{self.user_directory}/{self.test_filename}.java"
            
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write(processed_junit_code)
            
            # Store test file info
            self.test_file_path = test_file_path
            self.test_class_name = actual_class_name
            
            self.logger.info(f"✅ JUnit test file created from data: {test_file_path}")
            self.logger.info(f"📝 Test class name: {actual_class_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error creating JUnit test from data: {e}")
            return False

    def _process_code_for_package(self, java_code: str) -> str:
        """
        Process Java code to ensure proper package declaration consistency
        
        Args:
            java_code: Raw Java code
            
        Returns:
            str: Processed Java code without package declaration (FileCreator will add it)
        """
        try:
            # Remove any existing package declaration since FileCreator will add it
            java_code = re.sub(r'package\s+[^;]+;', '', java_code).strip()
            
            # Remove any leading newlines/whitespace
            java_code = java_code.lstrip()
            
            self.logger.info(f"📝 Processed Java code for package: {self.package_name}")
            return java_code
            
        except Exception as e:
            self.logger.error(f"❌ Error processing Java code: {e}")
            return java_code  # Return original code if processing fails

    def _extract_class_name_from_code(self, java_code: str) -> str:
        """
        Extract the actual class name from Java code
        
        Args:
            java_code: Java source code
            
        Returns:
            str: The actual class name found in the code
        """
        try:
            # Look for public class declaration
            public_class_match = re.search(r'public\s+class\s+(\w+)', java_code)
            if public_class_match:
                class_name = public_class_match.group(1)
                self.logger.info(f"🔍 Found public class: {class_name}")
                return class_name
            
            # Look for any class declaration
            class_match = re.search(r'class\s+(\w+)', java_code)
            if class_match:
                class_name = class_match.group(1)
                self.logger.info(f"🔍 Found class: {class_name}")
                return class_name
            
            self.logger.warning("⚠️ No class name found in code")
            return "UnknownClass"
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting class name: {e}")
            return "UnknownClass"

    def _process_junit_code_for_package(self, junit_code: str, class_name: str) -> tuple:
        """
        Process JUnit code to ensure proper package declaration
        
        Args:
            junit_code: Raw JUnit test code
            class_name: Class name for the test
            
        Returns:
            tuple: (processed_code, actual_class_name)
        """
        try:
            # Remove existing package declaration if any
            junit_code = re.sub(r'package\s+[^;]+;', '', junit_code).strip()
            
            # CRITICAL: Replace {{user_package}} placeholder with actual user package
            junit_code = junit_code.replace("{{user_package}}", self.user_directory)
            junit_code = junit_code.replace("{{USER_PACKAGE}}", self.user_directory)  # Handle uppercase variant
            self.logger.info(f"🔧 Replaced {{{{user_package}}}} with: {self.user_directory}")
            
            # Replace {{class_name}} placeholder if any
            junit_code = junit_code.replace("{{class_name}}", class_name)
            junit_code = junit_code.replace("{{CLASS_NAME}}", class_name)  # Handle uppercase variant
            self.logger.info(f"🔧 Replaced {{{{class_name}}}} with: {class_name}")
            
            # Extract actual class name from the processed code
            actual_class_name = self._extract_class_name_from_code(junit_code)
            
            # Add package declaration for user directory
            package_declaration = f"package {self.user_directory};\n\n"
            
            # Ensure proper imports
            if 'import org.junit' not in junit_code:
                import_statements = """import org.junit.Test;
import org.junit.Assert;
import static org.junit.Assert.*;

"""
                junit_code = import_statements + junit_code
            
            # Combine package + imports + test code
            processed_code = package_declaration + junit_code
            
            self.logger.info(f"📝 Processed JUnit code for package: {self.user_directory}, actual class: {actual_class_name}")
            return processed_code, actual_class_name
            
        except Exception as e:
            self.logger.error(f"❌ Error processing JUnit code: {e}")
            return junit_code, class_name  # Return original code if processing fails

    def create_file(self) -> subprocess.CompletedProcess:
        """
        Create Java file from user code dengan improved error handling
        Updated for new LLM workflow
        """
        try:
            self.check_and_create_dir()
            
            # Ensure we have a filename
            if not self.filename:
                self.filename = "UserCode"  # Default filename
                self.logger.warning("No class name detected, using default: UserCode")
                
            # Ensure package_name is set
            if not hasattr(self, 'package_name') or not self.package_name:
                self.package_name = self.user_directory
                
            self.logger.info(f"Creating Java file: {self.filename}.java in package {self.package_name}")
            
            # Process code to ensure proper package declaration
            processed_code = self._process_code_for_package(self.code)
            
            # Create main Java file with processed code
            user_path = os.path.join(self.config.main_directory, self.user_directory)
            os.makedirs(user_path, exist_ok=True)
            file_path = os.path.join(user_path, f"{self.filename}.java")
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(processed_code)
            
            self.logger.info(f"✅ Java file created successfully: {file_path}")
            
            # Setup JUnit test file if question_id is available
            if self.question_id:
                self.logger.info(f"Setting up JUnit test for question_id: {self.question_id}")
                junit_setup_success = self.setup_junit_test_file(self.question_id)
                if not junit_setup_success:
                    self.logger.warning("JUnit test setup failed, continuing without tests")
            
            # Compile the file and return compile result
            return self.compile_file()
                
        except Exception as e:
            self.logger.error(f"Error creating Java file: {e}")
            return self._create_error_result(f"Error creating Java file: {str(e)}")

    def compile_file(self) -> subprocess.CompletedProcess:
        """
        Compile Java file dengan improved error handling
        
        Returns:
            subprocess.CompletedProcess: Result of compilation
        """
        if not self.filename:
            self.logger.error("No filename available for compilation")
            return self._create_error_result("No Java file to compile")
            
        # Use working directory approach for consistent compilation
        if os.name == 'nt':  # Windows
            compile_command = f"cd {self.config.main_directory} && javac {self.user_directory}/{self.filename}.java"
        else:  # Linux/Unix  
            compile_command = f"cd {self.config.main_directory} && javac {self.user_directory}/{self.filename}.java"
        
        try:
            self.logger.info(f"Compiling: {self.filename}.java")
            self.logger.debug(f"Command: {compile_command}")
            
            output = subprocess.run(
                compile_command, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
            )
            
            if output.returncode == 0:
                self.logger.info("Compilation successful")
            else:
                error_msg = output.stderr.decode('utf-8', errors='ignore')
                self.logger.warning(f"Compilation failed: {error_msg}")
                # [NEW] Return the actual javac error in the output for better feedback
                return output
                
            return output
            
        except subprocess.TimeoutExpired:
            self.logger.error("Compilation timeout")
            return self._create_error_result("Compilation timeout")
        except Exception as e:
            self.logger.error(f"Compilation error: {e}")
            return self._create_error_result(f"Compilation error: {str(e)}")

    def run_file(self) -> subprocess.CompletedProcess:
        """
        Run compiled Java file dengan improved error handling
        
        Returns:
            subprocess.CompletedProcess: Result of program execution
        """
        if not self.filename:
            self.logger.error("No filename available for execution")
            return self._create_error_result("No Java file to run")
        
        # Build command with proper working directory and classpath
        user_dir_path = f"{self.config.main_directory}/{self.user_directory}"
        
        # Cross-platform compatible command
        if os.name == 'nt':  # Windows
            # Set working directory to parent of user directory and use fully qualified class name
            run_command = f"cd {self.config.main_directory} && java -cp . {self.user_directory}.{self.filename}"
        else:  # Linux/Unix
            run_command = f"cd {self.config.main_directory} && java -cp . {self.user_directory}.{self.filename}"
            
        try:
            self.logger.info(f"Running: {self.filename} with package {self.user_directory}")
            self.logger.debug(f"Command: {run_command}")
            
            # Provide dummy input to prevent blocking on Scanner
            dummy_input = "0\n1\ntest\n" * 10
            
            output = subprocess.run(
                run_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                input=dummy_input.encode('utf-8'),
                timeout=5  # 5 second timeout
            )
            
            if output.returncode == 0:
                self.logger.info("Program execution successful")
            else:
                error_msg = output.stderr.decode('utf-8')
                self.logger.warning(f"Program execution failed: {error_msg}")
                
            return output
            
        except subprocess.TimeoutExpired:
            self.logger.error("Program execution timeout")
            return self._create_error_result("Program execution timeout")
        except Exception as e:
            self.logger.error(f"Program execution error: {e}")
            return self._create_error_result(f"Program execution error: {str(e)}")

    def run_io_tests(self, test_cases: list) -> Dict[str, Any]:
        """
        Run I/O based tests by executing the program with inputs and checking outputs
        """
        results = []
        total_passed = 0
        total_cases = len(test_cases)
        
        if not self.filename:
            return {"score": 0, "logs": "No compiled file to run", "details": []}

        # Build command (reuse run_file logic)
        run_command = f"cd {self.config.main_directory} && java -cp . {self.user_directory}.{self.filename}"
        if os.name == 'nt':
            run_command = f"cd {self.config.main_directory} && java -cp . {self.user_directory}.{self.filename}"

        logs = []
        
        for i, case in enumerate(test_cases, 1):
            input_str = case.get('input', '')
            expected = case.get('expected_output', '')
            description = case.get('description', f'Test Case {i}')
            
            try:
                self.logger.info(f"Running Test {i}: {description}")
                logs.append(f"Running Test {i}: {description}")
                
                start_time = time.time()
                process = subprocess.run(
                    run_command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    input=input_str.encode('utf-8'),
                    timeout=3 # 3s timeout per case
                )
                duration = time.time() - start_time
                
                stdout = process.stdout.decode('utf-8', errors='ignore')
                stderr = process.stderr.decode('utf-8', errors='ignore')
                
                # Check output - SEMANTIC MATCHING
                passed = False
                
                # Normalize both strings
                def semantic_normalize(text):
                    # Remove non-alphanumeric except for decimals and common symbols
                    clean = re.sub(r'[^\w\s\.]', ' ', text)
                    # Convert to lowercase and join by single space
                    return " ".join(clean.lower().split())
                
                norm_stdout = semantic_normalize(stdout)
                norm_expected = semantic_normalize(expected)
                
                # Check if expected is in stdout or vice versa
                if norm_expected in norm_stdout or norm_stdout in norm_expected:
                    passed = True
                else:
                    # Fallback 2: Regex for numbers (check if all numbers in expected are in stdout)
                    expected_numbers = re.findall(r'\d+\.?\d*', norm_expected)
                    if expected_numbers and all(n in norm_stdout for n in expected_numbers):
                        passed = True
                        logs.append(f"⚠️ PASSED (Semantic Number Match)")
                    
                    # [PARTICIPATION GRADING] Fallback 3: Exit Code 0 = Pass (User Request)
                    elif process.returncode == 0:
                        passed = True
                        logs.append(f"⚠️ PASSED (Exit Code 0 - Participation Grading)")

                if passed:
                    total_passed += 1
                    logs.append(f"✅ PASSED")
                else:
                    logs.append(f"❌ FAILED. Expected: '{expected}'")
                    # Show simplified actual output (first 100 chars sanitized)
                    clean_actual_preview = " ".join(stdout.split())[:100]
                    logs.append(f"   Input: '{input_str.strip()}'") 
                    logs.append(f"   Actual: '{clean_actual_preview}...'")
                
                results.append({
                    "case": i,
                    "description": description,
                    "passed": passed,
                    "input": input_str,
                    "expected": expected,
                    "actual": stdout,
                    "duration": duration
                })
                
            except subprocess.TimeoutExpired:
                 logs.append(f"❌ TIMEOUT ( > 3s)")
                 results.append({
                    "case": i,
                    "description": description,
                    "passed": False,
                    "error": "Timeout"
                })
            except Exception as e:
                logs.append(f"❌ ERROR: {str(e)}")
                results.append({
                    "case": i,
                    "description": description,
                    "passed": False,
                    "error": str(e)
                })

        score = int((total_passed / total_cases) * 100) if total_cases > 0 else 0
        
        return {
            "score": score,
            "total_passed": total_passed,
            "total_cases": total_cases,
            "logs": "\n".join(logs),
            "details": results
        }

    def run_test(self) -> subprocess.CompletedProcess:
        """
        Run JUnit tests dengan improved error handling
        Updated for new test file system
        
        Returns:
            subprocess.CompletedProcess: Result of test execution
        """
        if not self.test_filename:
            self.logger.error("No test filename available")
            return self._create_error_result("No JUnit test file to run")
            
        # Build paths and commands with proper working directory
        # JUnit JAR files are in java_files directory - use absolute path from main directory
        if os.name == 'nt':  # Windows
            jar_path = f"{self.config.main_directory}junit-4.13.2.jar;{self.config.main_directory}hamcrest-core-1.3.jar"
        else:  # Linux/Unix
            jar_path = f"{self.config.main_directory}junit-4.13.2.jar:{self.config.main_directory}hamcrest-core-1.3.jar"
        
        # Compile test file first - using working directory approach
        if os.name == 'nt':  # Windows
            compile_test_command = (
                f"cd {self.config.main_directory} && "
                f"javac -cp \".;junit-4.13.2.jar;hamcrest-core-1.3.jar\" {self.user_directory}/{self.test_filename}.java"
            )
        else:  # Linux/Unix
            compile_test_command = (
                f"cd {self.config.main_directory} && "
                f"javac -cp \".:junit-4.13.2.jar:hamcrest-core-1.3.jar\" {self.user_directory}/{self.test_filename}.java"
            )
        
        # Run test command - using working directory approach  
        if os.name == 'nt':  # Windows
            run_test_command = (
                f"cd {self.config.main_directory} && "
                f"java -cp \".;junit-4.13.2.jar;hamcrest-core-1.3.jar\" org.junit.runner.JUnitCore {self.user_directory}.{self.test_filename}"
            )
        else:  # Linux/Unix
            run_test_command = (
                f"cd {self.config.main_directory} && "
                f"java -cp \".:junit-4.13.2.jar:hamcrest-core-1.3.jar\" org.junit.runner.JUnitCore {self.user_directory}.{self.test_filename}"
            )
        
        # DEBUG: Log paths and files for troubleshooting
        self.logger.info(f"JUnit Test Debug:")
        self.logger.info(f"   - Test filename: {self.test_filename}")
        self.logger.info(f"   - User directory: {self.user_directory}")
        self.logger.info(f"   - Compile command: {compile_test_command}")
        self.logger.info(f"   - Run command: {run_test_command}")
        
        # Check if test file exists
        test_file_path = f"{self.config.main_directory}/{self.user_directory}/{self.test_filename}.java"
        self.logger.info(f"   - Test file exists: {os.path.exists(test_file_path)}")
        if os.path.exists(test_file_path):
            self.logger.info(f"   - Test file size: {os.path.getsize(test_file_path)} bytes")
        
        # Check if JUnit JAR files exist
        junit_jar = f"{self.config.main_directory}junit-4.13.2.jar"
        hamcrest_jar = f"{self.config.main_directory}hamcrest-core-1.3.jar"
        self.logger.info(f"   - JUnit JAR exists: {os.path.exists(junit_jar)} at {junit_jar}")
        self.logger.info(f"   - Hamcrest JAR exists: {os.path.exists(hamcrest_jar)} at {hamcrest_jar}")
        
        try:
            # First compile the test file
            self.logger.info(f"Compiling test: {self.test_filename}.java")
            self.logger.debug(f"Command: {compile_test_command}")
            
            compile_result = subprocess.run(
                compile_test_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            
            if compile_result.returncode != 0:
                error_msg = compile_result.stderr.decode('utf-8')
                self.logger.error(f"❌ Test compilation failed: {error_msg}")
                return compile_result
            
            # Then run the test
            self.logger.info(f"Running JUnit tests: {self.test_filename}")
            self.logger.debug(f"Command: {run_test_command}")
            
            output = subprocess.run(
                run_test_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            
            if output.returncode == 0:
                self.logger.info("JUnit tests passed")
            else:
                error_msg = output.stderr.decode('utf-8')
                self.logger.warning(f"JUnit tests failed: {error_msg}")
                
            return output
            
        except subprocess.TimeoutExpired:
            self.logger.error("JUnit test execution timeout")
            return self._create_error_result("JUnit test execution timeout")
        except Exception as e:
            self.logger.error(f"JUnit test execution error: {e}")
            return self._create_error_result(f"JUnit test execution error: {str(e)}")

    def run_with_validation(self) -> Dict[str, Any]:
        """
        Enhanced run method with comprehensive validation
        Updated for new LLM workflow
        
        Returns:
            Dict: Complete execution results with validation and scoring
        """
    def run_with_validation(self) -> Dict[str, Any]:
        """
        Run code with validation using I/O Test Cases (Priority) or JUnit (Fallback)
        """
        try:
            self.logger.info(f"🚀 Starting execution for question_id: {self.question_id}")
            
            # 1. Create and compile
            create_n_compile = self.create_file()
            
            if create_n_compile.returncode != 0:
                error_msg = create_n_compile.stderr.decode("utf-8")
                self.logger.error(f"❌ Compilation failed: {error_msg}")
                return {
                    "java": error_msg, 
                    "test_output": "COMPILATION FAILED!",
                    "point": 0
                }

            # [NEW] 2. Check for I/O Test Cases (Priority)
            test_cases = []
            if self.question_data and isinstance(self.question_data, dict):
                 test_cases = self.question_data.get('test_cases', [])
            
            # If test_cases exists and is a non-empty list
            if test_cases and isinstance(test_cases, list) and len(test_cases) > 0:
                self.logger.info("🚀 Running I/O Based Grading...")
                io_results = self.run_io_tests(test_cases)
                
                score = io_results['score']
                logs = io_results['logs']
                
                return {
                    "java": logs, # Display test logs in output
                    "test_output": "TESTS COMPLETED",
                    "point": score,
                    "can_submit": score == 100,
                    "details": io_results.get('details', []) # [NEW] Pass details for AI Grading
                }

            # [FALLBACK] 3. JUnit Testing (Legacy support)
            self.logger.info("⚠️ No I/O cases found, checking for JUnit...")
            
            # Run program logic as usual if no IO tests
            final_output = self.run_file()
            
            if final_output.returncode != 0:
                error_msg = final_output.stderr.decode("utf-8")
                self.logger.error(f"Runtime error: {error_msg}")
                return {
                    "java": error_msg,
                    "test_output": "RUNTIME ERROR!",
                    "point": 0
                }
            
            java_output = final_output.stdout.decode("utf-8")
            
            # Get score from database (question_data) or use default
            database_score = 0
            if self.question_data and isinstance(self.question_data, dict):
                database_score = self.question_data.get('score', 0)
            
            if self.test_filename:
                self.logger.info("Running JUnit tests")
                junit_result = self.run_test()
                
                if junit_result.returncode == 0:
                    test_output = junit_result.stdout.decode("utf-8")
                    point = database_score
                    can_submit = True
                else:
                    test_output = junit_result.stderr.decode("utf-8") or junit_result.stdout.decode("utf-8")
                    point = 0
                    can_submit = False
            else:
                test_output = "Program executed successfully - no JUnit tests available"
                point = database_score
                can_submit = True
            
            return {
                "java": java_output,
                "test_output": test_output,
                "point": point,
                "can_submit": can_submit
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in run_with_validation: {e}")
            return {
                "java": f"System error: {str(e)}",
                "test_output": "SYSTEM ERROR!",
                "point": 0,
                "can_submit": False
            }

    def run(self) -> Dict[str, Any]:
        """
        Standard run method - simplified version
        For basic code execution without comprehensive validation
        """
        try:
            self.logger.info(f"🚀 Starting basic execution for user: {self.user_directory}")
            
            # Create and compile file
            create_n_compile = self.create_file()

            if create_n_compile.returncode != 0:
                error_msg = create_n_compile.stderr.decode("utf-8")
                return {
                    "java": error_msg, 
                    "test_output": "TEST FAILED!",  # Use old version message
                    "point": 0
                }

            # Run the main program
            final_output = self.run_file()

            # ADAPTED FROM LEGACY VERSION LOGIC
            test_output = None
            point = 0

            if final_output.returncode != 0:
                # Runtime error - return stderr like old version
                # Runtime error
                java = final_output.stderr.decode("utf-8")
                test_output = "RUNTIME ERROR!"
                point = 0
            else:
                java = final_output.stderr.decode("utf-8")
                if self.question_data and isinstance(self.question_data, dict):
                    database_score = self.question_data.get('score', 100)

                # Run JUnit tests like old version
                if self.test_filename:
                    junit_result = self.run_test()
                    if junit_result.returncode == 0:
                        # Tests passed = Full score
                        test_output = junit_result.stdout.decode("utf-8")
                        point = database_score
                        
                    else:
                        test_output = junit_result.stderr.decode("utf-8") or junit_result.stdout.decode("utf-8")
                        point = 0
                else:
                    # No tests available = Full score for execution
                    test_output = "Program executed successfully"
                    point = database_score

            return {
                "java": java,
                "test_output": test_output,
                "point": point
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in run method: {e}")
            return {
                "java": f"System error: {str(e)}",
                "test_output": "SYSTEM ERROR!",
                "point": 0
            }

    def validate_code_requirements(self) -> Optional[Dict[str, Any]]:
        """
        Validate code against specific requirements using CodeAnalyzer
        Returns detailed validation results
        """
        if not self.question_data:
            self.logger.warning("No question_data available for validation")
            return None
            
        try:
            # Get actual program output by running the code
            actual_output = None
            try:
                run_result = self.run_file()
                if run_result.returncode == 0:
                    actual_output = run_result.stdout.decode("utf-8")
                    self.logger.debug(f"Captured actual output for validation: {repr(actual_output[:100])}")
                else:
                    self.logger.warning(f"Program execution failed during validation: {run_result.stderr}")
            except Exception as e:
                self.logger.warning(f"Failed to capture actual output for validation: {e}")
            
            # Pass actual output to analyzer
            analyzer = CodeAnalyzer(self.code, self.question_data, actual_output)
            validation_result = analyzer.calculate_compliance_score()
            
            if validation_result:
                self.logger.info(f"✅ Validation completed with score: {validation_result.get('scores', {})}")
            else:
                self.logger.warning("⚠️ Validation returned no results")
                
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Error in code validation: {str(e)}")
            return None

    def _create_error_result(self, error_message: str):
        """Create a mock error result object"""
        return type('ErrorResult', (), {
            'returncode': 1, 
            'stderr': error_message.encode('utf-8'), 
            'stdout': b''
        })()

    # Legacy methods for backward compatibility
    def copy_admin_files_to_user(self):
        """Legacy method - kept for backward compatibility"""
        self.logger.warning("Using legacy copy_admin_files_to_user - consider updating to new workflow")
        return False

    def copy_admin_files_to_user_by_classname(self):
        """Legacy method - kept for backward compatibility"""
        self.logger.warning("Using legacy copy_admin_files_to_user_by_classname - consider updating to new workflow")
        return False
