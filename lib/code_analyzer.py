import re
import ast
import subprocess
import tempfile
import os
from typing import Dict, List, Any, Optional

class CodeAnalyzer:
    """
    Analyzer untuk menganalisis kode Java dan memvalidasi implementasi
    terhadap requirement yang spesifik
    """
    
    def __init__(self, code: str, requirements: Dict[str, Any], actual_output: str = None):
        self.code = code
        self.requirements = requirements
        self.actual_output = actual_output  # Add actual program output
        self.analysis_results = {}
        
    def analyze_variable_declarations(self) -> Dict[str, Any]:
        """Analyze variable declarations in the code"""
        results = {
            'declared_variables': {},
            'missing_variables': [],
            'incorrect_values': [],
            'correct_variables': []
        }
        
        # Required variables from requirements
        required_vars = self.requirements.get('variable_requirements', [])
        
        # Patterns untuk berbagai tipe variabel
        patterns = {
            'int': r'int\s+(\w+)\s*=\s*(\d+)',
            'double': r'double\s+(\w+)\s*=\s*([\d.]+)',
            'float': r'float\s+(\w+)\s*=\s*([\d.]+)f?',
            'String': r'String\s+(\w+)\s*=\s*"([^"]*)"',
            'long': r'long\s+(\w+)\s*=\s*(\d+)L?',
            'boolean': r'boolean\s+(\w+)\s*=\s*(true|false)'
        }
        
        # Extract all variable declarations from code
        for var_type, pattern in patterns.items():
            matches = re.findall(pattern, self.code, re.IGNORECASE)
            for match in matches:
                var_name, var_value = match
                results['declared_variables'][var_name] = {
                    'type': var_type,
                    'value': var_value,
                    'declared': True
                }
        
        # Check against requirements
        for req_var in required_vars:
            var_name = req_var.get('name')
            expected_value = str(req_var.get('expected_value'))
            var_type = req_var.get('type', 'int')
            
            if var_name in results['declared_variables']:
                actual_value = str(results['declared_variables'][var_name]['value'])
                if actual_value == expected_value:
                    results['correct_variables'].append({
                        'name': var_name,
                        'expected': expected_value,
                        'actual': actual_value,
                        'type': var_type
                    })
                else:
                    results['incorrect_values'].append({
                        'name': var_name,
                        'expected': expected_value,
                        'actual': actual_value,
                        'type': var_type
                    })
            else:
                results['missing_variables'].append({
                    'name': var_name,
                    'expected': expected_value,
                    'type': var_type
                })
        
        return results
    
    def analyze_calculations(self) -> Dict[str, Any]:
        """Analyze calculations and formulas in the code"""
        results = {
            'calculation_expressions': [],
            'correct_formulas': [],
            'incorrect_formulas': [],
            'missing_calculations': []
        }
        
        # Required calculations from requirements
        required_calculations = self.requirements.get('calculation_requirements', [])
        
        # Pattern untuk mencari assignment dengan operasi matematika
        calc_pattern = r'(\w+)\s*=\s*([^;]+);'
        matches = re.findall(calc_pattern, self.code)
        
        for match in matches:
            var_name, expression = match
            expression = expression.strip()
            # Skip simple assignments without calculations
            if any(op in expression for op in ['+', '-', '*', '/', '%']):
                results['calculation_expressions'].append({
                    'variable': var_name,
                    'expression': expression
                })
        
        # Validate against requirements
        for req_calc in required_calculations:
            formula_pattern = req_calc.get('pattern', '')
            result_var = req_calc.get('result_variable', '')
            
            # Check if the required calculation pattern exists
            if formula_pattern and any(formula_pattern in expr['expression'] 
                                     for expr in results['calculation_expressions'] 
                                     if expr['variable'] == result_var):
                results['correct_formulas'].append(req_calc)
            else:
                results['incorrect_formulas'].append(req_calc)
        
        return results
    
    def analyze_output_statements(self) -> Dict[str, Any]:
        """Analyze output against actual program output if available, otherwise analyze source code"""
        results = {
            'print_statements': [],
            'correct_outputs': [],
            'missing_outputs': []
        }
        
        # Required outputs from requirements
        required_outputs = self.requirements.get('required_outputs', [])
        
        if self.actual_output:
            # Use actual program output for validation (BETTER APPROACH)
            print(f"DEBUG: Analyzing actual output: {repr(self.actual_output[:200])}")
            
            for req_output in required_outputs:
                output_text = req_output.get('text', '')
                print(f"DEBUG: Looking for: {repr(output_text)}")
                
                # Check if required output exists in actual output
                if output_text in self.actual_output:
                    results['correct_outputs'].append(req_output)
                    print(f"DEBUG: Found output: {output_text}")
                else:
                    results['missing_outputs'].append(req_output)
                    print(f"DEBUG: Missing output: {output_text}")
        else:
            # Fallback: analyze source code print statements  
            print("DEBUG: No actual output provided, analyzing source code")
            
            # Pattern untuk System.out.println
            print_pattern = r'System\.out\.println\s*\(\s*([^)]+)\s*\)'
            matches = re.findall(print_pattern, self.code)
            
            for match in matches:
                results['print_statements'].append(match.strip())
            
            for req_output in required_outputs:
                output_text = req_output.get('text', '')
                if any(output_text in stmt for stmt in results['print_statements']):
                    results['correct_outputs'].append(req_output)
                else:
                    results['missing_outputs'].append(req_output)
        
        return results
    
    def calculate_compliance_score(self) -> Dict[str, Any]:
        """Calculate compliance score based on analysis"""
        var_analysis = self.analyze_variable_declarations()
        calc_analysis = self.analyze_calculations()
        output_analysis = self.analyze_output_statements()
        
        # Calculate scores
        total_required_vars = len(self.requirements.get('variable_requirements', []))
        correct_vars = len(var_analysis['correct_variables'])
        incorrect_vars = len(var_analysis['incorrect_values'])
        missing_vars = len(var_analysis['missing_variables'])
        
        total_required_calcs = len(self.requirements.get('calculation_requirements', []))
        correct_calcs = len(calc_analysis['correct_formulas'])
        
        total_required_outputs = len(self.requirements.get('required_outputs', []))
        correct_outputs = len(output_analysis['correct_outputs'])
        
        # Calculate percentages
        var_score = (correct_vars / total_required_vars * 100) if total_required_vars > 0 else 100
        calc_score = (correct_calcs / total_required_calcs * 100) if total_required_calcs > 0 else 100
        output_score = (correct_outputs / total_required_outputs * 100) if total_required_outputs > 0 else 100
        
        # Overall score (weighted average)
        overall_score = (var_score * 0.4 + calc_score * 0.4 + output_score * 0.2)
        
        return {
            'variable_analysis': var_analysis,
            'calculation_analysis': calc_analysis,
            'output_analysis': output_analysis,
            'scores': {
                'variable_score': var_score,
                'calculation_score': calc_score,
                'output_score': output_score,
                'overall_score': overall_score
            },
            'compliance_level': self._get_compliance_level(overall_score),
            'detailed_feedback': self._generate_detailed_feedback(
                var_analysis, calc_analysis, output_analysis
            )
        }
    
    def _get_compliance_level(self, score: float) -> str:
        """Get compliance level based on score"""
        if score >= 90:
            return "EXCELLENT"
        elif score >= 80:
            return "GOOD"
        elif score >= 70:
            return "ACCEPTABLE"
        elif score >= 60:
            return "NEEDS_IMPROVEMENT"
        else:
            return "POOR"
    
    def _generate_detailed_feedback(self, var_analysis: Dict, calc_analysis: Dict, 
                                  output_analysis: Dict) -> List[str]:
        """Generate detailed feedback for the student"""
        feedback = []
        
        # Variable feedback
        if var_analysis['incorrect_values']:
            for var in var_analysis['incorrect_values']:
                feedback.append(f"❌ Variable '{var['name']}' has incorrect value. "
                              f"Expected: {var['expected']}, Found: {var['actual']}")
        
        if var_analysis['missing_variables']:
            for var in var_analysis['missing_variables']:
                feedback.append(f"❌ Missing required variable '{var['name']}' "
                              f"with value {var['expected']}")
        
        if var_analysis['correct_variables']:
            for var in var_analysis['correct_variables']:
                feedback.append(f"✅ Variable '{var['name']}' correctly set to {var['expected']}")
        
        # Calculation feedback
        if calc_analysis['incorrect_formulas']:
            for calc in calc_analysis['incorrect_formulas']:
                feedback.append(f"❌ Missing or incorrect calculation for "
                              f"'{calc.get('result_variable', 'unknown')}'")
        
        # Output feedback
        if output_analysis['missing_outputs']:
            for output in output_analysis['missing_outputs']:
                feedback.append(f"❌ Missing required output: '{output.get('text', '')}'")
        
        return feedback

class ValidationResult:
    """Class to represent validation results"""
    
    def __init__(self, passed: bool, score: float, feedback: List[str], 
                 detailed_analysis: Dict[str, Any]):
        self.passed = passed
        self.score = score
        self.feedback = feedback
        self.detailed_analysis = detailed_analysis
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'passed': self.passed,
            'score': self.score,
            'feedback': self.feedback,
            'detailed_analysis': self.detailed_analysis
        }
