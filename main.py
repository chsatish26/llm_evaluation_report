"""
Unified LLM Evaluation Framework
Supports AWS Bedrock and OpenAI models with comprehensive evaluation
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

from llm_clients import LLMClientFactory
from evaluators import UnifiedEvaluator
from report_generator import PDFReportGenerator
from deepeval_integration import DeepEvalIntegration

load_dotenv()


class LLMEvaluationFramework:
    """Main evaluation framework supporting multiple LLM providers"""
    
    def __init__(self):
        self.config = self._load_config()
        self.templates = self._load_templates()
        self.results = []
        self.log_file = self._setup_logging()
        
        # Initialize components
        self.deepeval = DeepEvalIntegration()
        self.report_generator = PDFReportGenerator()
        
        print(f"\n{'='*70}")
        print(f"LLM EVALUATION FRAMEWORK INITIALIZED")
        print(f"{'='*70}")
        print(f"Test Models: {self.config['test_models']}")
        print(f"Judge Model: {self.config['judge_model']}")
        print(f"DeepEval: {'Enabled' if self.deepeval.enabled else 'Disabled'}")
        print(f"Log File: {self.log_file}")
        print(f"{'='*70}\n")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment"""
        
        # Parse test models from EVAL_MODEL_MAPPING
        test_models = []
        model_mapping = os.getenv('EVAL_MODEL_MAPPING', '')
        if model_mapping:
            for mapping in model_mapping.split(','):
                if ':' in mapping:
                    provider, model_id = mapping.strip().split(':', 1)
                    test_models.append({
                        'provider': provider.strip(),
                        'model_id': model_id.strip()
                    })
        
        # Fallback to single model
        if not test_models:
            test_models = [{
                'provider': os.getenv('PROVIDER', 'aws'),
                'model_id': os.getenv('MODEL_ID', 'us.anthropic.claude-3-5-sonnet-20240620-v1:0')
            }]
        
        return {
            'test_models': test_models,
            'judge_model': {
                'provider': os.getenv('JUDGE_PROVIDER', 'aws'),
                'model_id': os.getenv('JUDGE_MODEL_ID', 'us.anthropic.claude-3-5-sonnet-20240620-v1:0')
            },
            'template_path': os.getenv('PROMPT_TEMPLATE_PATH', 'prompt_templates_enhanced_deepeval.json'),
            'output_path': os.getenv('OUTPUT_REPORT_PATH', 'evaluation_reports'),
            'max_tokens': int(os.getenv('MAX_TOKENS', '4096')),
            'temperature': float(os.getenv('TEMPERATURE', '1.0'))
        }
    
    def _load_templates(self) -> Dict[str, List[Dict]]:
        """Load evaluation templates"""
        try:
            with open(self.config['template_path'], 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Template file not found, using defaults")
            return self._get_default_templates()
    
    def _get_default_templates(self) -> Dict[str, List[Dict]]:
        """Default templates if file not found"""
        return {
            "faithfulness": [{
                "name": "Basic Factual Test",
                "prompt": "What is the capital of France?",
                "evaluation_method": "comprehensive",
                "expected_keywords": ["Paris"],
                "context": ["France is a country in Western Europe"]
            }]
        }
    
    def _setup_logging(self) -> str:
        """Setup logging file"""
        Path(self.config['output_path']).mkdir(parents=True, exist_ok=True)
        log_file = os.path.join(self.config['output_path'], 'evaluation_logs.jsonl')
        return log_file
    
    def run_evaluation(self, categories: List[str] = None) -> Dict[str, List]:
        """Run evaluation across all configured models"""
        
        all_results = {}
        categories = categories or list(self.templates.keys())
        
        for model_config in self.config['test_models']:
            print(f"\n{'='*60}")
            print(f"Testing: {model_config['provider']}/{model_config['model_id']}")
            print(f"{'='*60}")
            
            model_results = self._evaluate_single_model(model_config, categories)
            model_key = f"{model_config['provider']}:{model_config['model_id']}"
            all_results[model_key] = model_results
            self.results.extend(model_results)
            
            # Print summary
            self._print_model_summary(model_results, model_config)
        
        return all_results
    
    def _evaluate_single_model(self, model_config: Dict, categories: List[str]) -> List:
        """Evaluate a single model across categories"""
        
        # Create clients
        test_client = LLMClientFactory.create_client(
            provider=model_config['provider'],
            model_id=model_config['model_id'],
            max_tokens=self.config['max_tokens'],
            temperature=self.config['temperature']
        )
        
        judge_client = LLMClientFactory.create_client(
            provider=self.config['judge_model']['provider'],
            model_id=self.config['judge_model']['model_id'],
            is_judge=True
        )
        
        evaluator = UnifiedEvaluator(
            test_client=test_client,
            judge_client=judge_client,
            deepeval=self.deepeval,
            log_file=self.log_file
        )
        
        results = []
        
        for category in categories:
            templates = self.templates.get(category, [])
            
            if not templates:
                continue
            
            print(f"\nCategory: {category.upper()} ({len(templates)} tests)")
            
            for idx, template in enumerate(templates, 1):
                template['category'] = category
                print(f"  Test {idx}/{len(templates)}: {template['name']}")
                
                # Run evaluation
                result = evaluator.evaluate_test(
                    prompt=template['prompt'],
                    template=template,
                    model_info=model_config
                )
                
                results.append(result)
                
                # Print result
                status = "PASS" if result['status'] == "PASS" else "FAIL" if result['status'] == "FAIL" else "WARN"
                print(f"    [{status}] Score: {result['overall_score']:.2f} | Risk: {result['risk_level']}")
        
        return results
    
    def _print_model_summary(self, results: List[Dict], model_config: Dict):
        """Print summary for a model"""
        total = len(results)
        passed = sum(1 for r in results if r['status'] == 'PASS')
        failed = sum(1 for r in results if r['status'] == 'FAIL')
        
        avg_score = sum(r['overall_score'] for r in results) / total if total > 0 else 0
        
        print(f"\nModel Summary:")
        print(f"  Total: {total} | Passed: {passed} | Failed: {failed}")
        print(f"  Pass Rate: {(passed/total*100):.1f}% | Avg Score: {avg_score:.2f}")
    
    def generate_reports(self) -> Dict[str, str]:
        """Generate all reports"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        reports = {}
        
        # PDF Report
        pdf_path = os.path.join(
            self.config['output_path'],
            f"llm_evaluation_report_{timestamp}.pdf"
        )
        
        framework_info = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'test_models': [f"{m['provider']}/{m['model_id']}" for m in self.config['test_models']],
            'judge_model': f"{self.config['judge_model']['provider']}/{self.config['judge_model']['model_id']}",
            'deepeval_enabled': self.deepeval.enabled,
            'deepeval_metrics': self.deepeval.metrics if self.deepeval.enabled else []
        }
        
        self.report_generator.generate_report(
            results=self.results,
            framework_info=framework_info,
            output_path=pdf_path
        )
        
        reports['pdf'] = pdf_path
        
        # JSON Export
        json_path = os.path.join(
            self.config['output_path'],
            f"llm_evaluation_data_{timestamp}.json"
        )
        
        with open(json_path, 'w') as f:
            json.dump({
                'metadata': framework_info,
                'results': self.results
            }, f, indent=2, default=str)
        
        reports['json'] = json_path
        
        return reports


def main():
    """Main execution"""
    
    print("Starting LLM Evaluation Framework...")
    
    # Initialize framework
    framework = LLMEvaluationFramework()
    
    # Run evaluation
    results = framework.run_evaluation()
    
    # Generate reports
    reports = framework.generate_reports()
    
    # Print final summary
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total Tests: {len(framework.results)}")
    print(f"Reports Generated:")
    for report_type, path in reports.items():
        print(f"  {report_type.upper()}: {path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()