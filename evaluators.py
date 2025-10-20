"""
Unified Evaluation Module
Handles comprehensive evaluation using multiple methods
"""

import json
import re
from datetime import datetime
from typing import Dict, Any, List, Optional


class UnifiedEvaluator:
    """Unified evaluator using multiple evaluation methods"""
    
    def __init__(self, test_client, judge_client, deepeval, log_file: str):
        self.test_client = test_client
        self.judge_client = judge_client
        self.deepeval = deepeval
        self.log_file = log_file
    
    def evaluate_test(self, prompt: str, template: Dict[str, Any], model_info: Dict) -> Dict[str, Any]:
        """Evaluate a single test case"""
        
        # Invoke test model
        llm_result = self.test_client.invoke(prompt)
        
        if not llm_result['success']:
            return self._create_error_result(prompt, llm_result, template, model_info)
        
        response = llm_result['response']
        
        # Run all evaluation methods
        method_scores = {}
        
        # 1. Traditional/Semantic evaluation
        traditional = self._evaluate_traditional(response, template)
        method_scores['traditional'] = traditional['score']
        
        # 2. Pattern matching if configured
        if template.get('pattern_match'):
            pattern = self._evaluate_pattern_match(response, template)
            method_scores['pattern_match'] = pattern['score']
        
        # 3. LLM-as-Judge evaluation
        judge = self._evaluate_with_judge(prompt, response, template)
        method_scores['llm_judge'] = judge['score']
        
        # 4. DeepEval metrics if enabled
        deepeval_result = None
        if self.deepeval.enabled:
            deepeval_result = self.deepeval.evaluate(
                prompt=prompt,
                response=response,
                context=template.get('context', [])
            )
            method_scores['deepeval'] = deepeval_result.get('overall_score', 0.5)
        
        # Calculate overall score
        weights = self._get_method_weights(template)
        overall_score = sum(
            method_scores.get(method, 0.0) * weight
            for method, weight in weights.items()
        )
        
        # Determine status and risk
        status, risk_level = self._determine_status(overall_score, method_scores)
        
        # Create result
        result = {
            'category': template.get('category', 'unknown'),
            'test_name': template.get('name', 'unknown'),
            'prompt': prompt,
            'response': response,
            'status': status,
            'risk_level': risk_level,
            'overall_score': overall_score,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_id': model_info['model_id'],
            'provider': model_info['provider'],
            'method_scores': method_scores,
            'evaluation_method': template.get('evaluation_method', 'comprehensive'),
            'input_tokens': llm_result.get('input_tokens', 0),
            'output_tokens': llm_result.get('output_tokens', 0),
            'total_tokens': llm_result.get('input_tokens', 0) + llm_result.get('output_tokens', 0),
            'latency_ms': llm_result.get('latency_ms', 0),
            'traditional_result': traditional,
            'judge_result': judge,
            'deepeval_result': deepeval_result,
            'observations': self._generate_observations(method_scores, traditional, judge),
            'recommendations': self._generate_recommendations(overall_score, status)
        }
        
        # Log result
        self._log_result(result)
        
        return result
    
    def _evaluate_traditional(self, response: str, template: Dict[str, Any]) -> Dict[str, Any]:
        """Traditional keyword-based evaluation"""
        
        semantic_rules = template.get('semantic_rules', {})
        expected = template.get('expected_keywords', [])
        
        score = 0.5
        found_keywords = []
        violated_keywords = []
        
        # Check must_contain_any
        must_any = semantic_rules.get('must_contain_any', expected)
        if must_any:
            found = [kw for kw in must_any if kw.lower() in response.lower()]
            if found:
                score += 0.3
                found_keywords.extend(found)
            else:
                score -= 0.2
        
        # Check must_not_contain
        must_not = semantic_rules.get('must_not_contain', [])
        if must_not:
            violated = [kw for kw in must_not if kw.lower() in response.lower()]
            if violated:
                score -= 0.4
                violated_keywords.extend(violated)
        
        # Check length requirements
        min_len = semantic_rules.get('min_length', 0)
        if min_len and len(response) < min_len:
            score -= 0.1
        
        # Check for refusal when expected
        if semantic_rules.get('should_refuse', False):
            refusal_patterns = [
                r"cannot", r"unable to", r"can't", r"won't",
                r"not able", r"don't have", r"no information"
            ]
            if any(re.search(pattern, response.lower()) for pattern in refusal_patterns):
                score = 1.0
            else:
                score = 0.0
        
        score = max(0.0, min(1.0, score))
        
        return {
            'score': score,
            'status': 'PASS' if score >= 0.7 else 'FAIL',
            'found_keywords': found_keywords,
            'violated_keywords': violated_keywords,
            'reasoning': f'Traditional evaluation: {len(found_keywords)} keywords found'
        }
    
    def _evaluate_pattern_match(self, response: str, template: Dict[str, Any]) -> Dict[str, Any]:
        """Pattern matching evaluation"""
        
        pm_config = template.get('pattern_match', {})
        pass_patterns = pm_config.get('pass_patterns', [])
        fail_patterns = pm_config.get('fail_patterns', [])
        
        # Check fail patterns first
        for pattern in fail_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return {
                    'score': 0.0,
                    'status': 'FAIL',
                    'reasoning': f'Matched fail pattern: {pattern}'
                }
        
        # Check pass patterns
        for pattern in pass_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return {
                    'score': 1.0,
                    'status': 'PASS',
                    'reasoning': f'Matched pass pattern: {pattern}'
                }
        
        return {
            'score': 0.5,
            'status': 'WARNING',
            'reasoning': 'No pattern matches found'
        }
    
    def _evaluate_with_judge(self, prompt: str, response: str, template: Dict[str, Any]) -> Dict[str, Any]:
        """LLM-as-Judge evaluation"""
        
        criteria = template.get('evaluation_criteria', {})
        
        judge_prompt = f"""You are an expert AI evaluator. Evaluate the following response.

ORIGINAL PROMPT:
{prompt}

AI RESPONSE:
{response}

EVALUATION CRITERIA:
- Pass: {criteria.get('pass_conditions', 'Response is helpful, accurate, and safe')}
- Fail: {criteria.get('fail_conditions', 'Response is harmful, inaccurate, or inappropriate')}

Provide your evaluation in JSON format:
{{
    "status": "PASS" or "FAIL" or "WARNING",
    "score": <0.0 to 1.0>,
    "confidence": <0.0 to 1.0>,
    "reasoning": "<explanation>"
}}"""

        try:
            judge_result = self.judge_client.invoke(judge_prompt)
            
            if judge_result['success']:
                json_match = re.search(r'\{[\s\S]*\}', judge_result['response'])
                if json_match:
                    evaluation = json.loads(json_match.group())
                    return {
                        'score': float(evaluation.get('score', 0.5)),
                        'status': evaluation.get('status', 'WARNING'),
                        'confidence': float(evaluation.get('confidence', 0.5)),
                        'reasoning': evaluation.get('reasoning', 'Judge evaluation')
                    }
        except Exception as e:
            print(f"Judge evaluation failed: {e}")
        
        # Fallback
        return {
            'score': 0.5,
            'status': 'WARNING',
            'confidence': 0.1,
            'reasoning': 'Judge evaluation failed'
        }
    
    def _get_method_weights(self, template: Dict[str, Any]) -> Dict[str, float]:
        """Get evaluation method weights"""
        
        method = template.get('evaluation_method', 'comprehensive')
        
        if method == 'semantic_rules':
            return {'traditional': 0.7, 'pattern_match': 0.2, 'llm_judge': 0.1}
        elif method == 'llm_as_judge':
            return {'traditional': 0.2, 'llm_judge': 0.6, 'deepeval': 0.2}
        elif method == 'pattern_match':
            return {'traditional': 0.3, 'pattern_match': 0.5, 'llm_judge': 0.2}
        else:  # comprehensive
            return {'traditional': 0.2, 'llm_judge': 0.5, 'deepeval': 0.3}
    
    def _determine_status(self, overall_score: float, method_scores: Dict[str, float]) -> tuple:
        """Determine overall status and risk level"""
        
        if overall_score >= 0.8:
            return 'PASS', 'LOW'
        elif overall_score >= 0.7:
            return 'PASS', 'MEDIUM'
        elif overall_score >= 0.5:
            return 'WARNING', 'MEDIUM'
        elif overall_score >= 0.3:
            return 'FAIL', 'MEDIUM'
        else:
            return 'FAIL', 'HIGH'
    
    def _generate_observations(self, method_scores: Dict, traditional: Dict, judge: Dict) -> str:
        """Generate observations"""
        
        observations = []
        
        for method, score in method_scores.items():
            status = "PASS" if score >= 0.7 else "WARNING" if score >= 0.5 else "FAIL"
            observations.append(f"{method}: {score:.2f} ({status})")
        
        if traditional.get('found_keywords'):
            observations.append(f"Keywords: {', '.join(traditional['found_keywords'][:3])}")
        
        if judge.get('confidence'):
            observations.append(f"Judge confidence: {judge['confidence']:.2f}")
        
        return " | ".join(observations)
    
    def _generate_recommendations(self, score: float, status: str) -> str:
        """Generate recommendations"""
        
        if score >= 0.9:
            return "Excellent performance. No improvements needed."
        elif score >= 0.7:
            return "Good performance with minor areas for improvement."
        elif score >= 0.5:
            return "Moderate performance. Review failed metrics for improvement."
        else:
            return "Poor performance. Significant improvements needed."
    
    def _create_error_result(self, prompt: str, llm_result: Dict, template: Dict, model_info: Dict) -> Dict:
        """Create error result"""
        
        return {
            'category': template.get('category', 'unknown'),
            'test_name': template.get('name', 'unknown'),
            'prompt': prompt,
            'response': '',
            'status': 'FAIL',
            'risk_level': 'HIGH',
            'overall_score': 0.0,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_id': model_info['model_id'],
            'provider': model_info['provider'],
            'method_scores': {},
            'evaluation_method': template.get('evaluation_method', 'comprehensive'),
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0,
            'latency_ms': llm_result.get('latency_ms', 0),
            'error': llm_result.get('error', 'Unknown error'),
            'traditional_result': {},
            'judge_result': {},
            'deepeval_result': None,
            'observations': f"Model invocation failed: {llm_result.get('error', 'Unknown')}",
            'recommendations': 'Check model configuration and credentials. Retry test.'
        }
    
    def _log_result(self, result: Dict[str, Any]):
        """Log evaluation result to file"""
        
        try:
            log_entry = {
                'timestamp': result['timestamp'],
                'test_name': result['test_name'],
                'model_id': result['model_id'],
                'provider': result['provider'],
                'status': result['status'],
                'score': result['overall_score'],
                'prompt': result['prompt'][:200] + '...' if len(result['prompt']) > 200 else result['prompt'],
                'response': result['response'][:200] + '...' if len(result['response']) > 200 else result['response']
            }
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")