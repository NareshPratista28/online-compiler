import re
import logging
from typing import Optional, List, Dict, Any

class CodeParser:
    """
    Java code parsing utilities for LLM-generated code
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_class_name(self, code: str) -> Optional[str]:
        """Extract class name from Java code"""
        if not code:
            self.logger.warning("Empty code provided for class name extraction")
            return None
            
        # Look for public class declaration first
        public_class_pattern = r'public\s+class\s+(\w+)'
        match = re.search(public_class_pattern, code)
        if match:
            class_name = match.group(1)
            self.logger.debug(f"Found public class: {class_name}")
            return class_name
            
        # Fallback: look for any class declaration
        class_pattern = r'class\s+(\w+)'
        match = re.search(class_pattern, code)
        if match:
            class_name = match.group(1)
            self.logger.debug(f"Found class: {class_name}")
            return class_name
            
        self.logger.warning("No class name found in code")
        return None
    
    def extract_package_declaration(self, code: str) -> Optional[str]:
        """Extract package declaration from Java code"""
        if not code:
            return None
            
        package_pattern = r'package\s+([\w.]+)\s*;'
        match = re.search(package_pattern, code)
        if match:
            package_name = match.group(1)
            self.logger.debug(f"Found package: {package_name}")
            return package_name
            
        return None
    
    def extract_imports(self, code: str) -> List[str]:
        """Extract import statements from Java code"""
        if not code:
            return []
            
        import_pattern = r'import\s+([\w.*]+)\s*;'
        imports = re.findall(import_pattern, code)
        self.logger.debug(f"Found {len(imports)} imports")
        return imports
    
    def add_package_declaration(self, code: str, package_name: str) -> str:
        """Add or replace package declaration in Java code"""
        if not code or not package_name:
            return code
            
        # Remove existing package declaration if present
        code_without_package = re.sub(r'package\s+[\w.]+\s*;\s*\n?', '', code)
        
        # Add new package declaration at the beginning
        package_declaration = f"package {package_name};\n\n"
        
        # Find the first import or class declaration to insert package before it
        lines = code_without_package.split('\n')
        insert_index = 0
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if stripped_line.startswith('import ') or stripped_line.startswith('public class ') or stripped_line.startswith('class '):
                insert_index = i
                break
        
        # Insert package declaration
        lines.insert(insert_index, package_declaration.rstrip())
        
        result = '\n'.join(lines)
        self.logger.debug(f"Added package declaration: {package_name}")
        return result
    
    def replace_package_placeholder(self, code: str, package_name: str) -> str:
        """Replace {{user_package}} placeholder with actual package name"""
        if not code or not package_name:
            return code
            
        # Replace placeholder
        result = code.replace("{{user_package}}", package_name)
        
        if "{{user_package}}" in code:
            self.logger.debug(f"Replaced {{user_package}} with {package_name}")
        
        return result
    
    def validate_java_syntax_basic(self, code: str) -> Dict[str, Any]:
        """Basic Java syntax validation"""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        if not code:
            validation_result["is_valid"] = False
            validation_result["errors"].append("Empty code")
            return validation_result
        
        # Check for basic Java structure
        if not re.search(r'class\s+\w+', code):
            validation_result["errors"].append("No class declaration found")
            validation_result["is_valid"] = False
        
        # Check for balanced braces
        open_braces = code.count('{')
        close_braces = code.count('}')
        if open_braces != close_braces:
            validation_result["errors"].append(f"Unbalanced braces: {open_braces} open, {close_braces} close")
            validation_result["is_valid"] = False
        
        # Check for balanced parentheses
        open_parens = code.count('(')
        close_parens = code.count(')')
        if open_parens != close_parens:
            validation_result["errors"].append(f"Unbalanced parentheses: {open_parens} open, {close_parens} close")
            validation_result["is_valid"] = False
        
        # Check for placeholder presence (should have some)
        if "..." not in code:
            validation_result["warnings"].append("No placeholder '...' found - code might be complete")
        
        self.logger.debug(f"Java syntax validation: {'PASS' if validation_result['is_valid'] else 'FAIL'}")
        return validation_result
    
    def extract_method_signatures(self, code: str) -> List[Dict[str, str]]:
        """Extract method signatures from Java code"""
        if not code:
            return []
        
        # Pattern to match method declarations
        method_pattern = r'(public|private|protected)?\s*(static)?\s*(\w+)\s+(\w+)\s*\([^)]*\)'
        methods = []
        
        for match in re.finditer(method_pattern, code):
            visibility = match.group(1) or "default"
            is_static = match.group(2) is not None
            return_type = match.group(3)
            method_name = match.group(4)
            
            methods.append({
                "visibility": visibility,
                "is_static": is_static,
                "return_type": return_type,
                "method_name": method_name,
                "full_signature": match.group(0)
            })
        
        self.logger.debug(f"Found {len(methods)} method signatures")
        return methods
    
    def count_placeholders(self, code: str) -> int:
        """Count the number of '...' placeholders in code"""
        if not code:
            return 0
        
        count = code.count("...")
        self.logger.debug(f"Found {count} placeholders in code")
        return count
    
    def has_main_method(self, code: str) -> bool:
        """Check if code has a main method"""
        if not code:
            return False
        
        main_pattern = r'public\s+static\s+void\s+main\s*\(\s*String\s*\[\s*\]\s*\w+\s*\)'
        has_main = bool(re.search(main_pattern, code))
        
        self.logger.debug(f"Main method found: {has_main}")
        return has_main
