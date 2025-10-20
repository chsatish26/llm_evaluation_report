"""
Unified LLM Clients for AWS Bedrock and OpenAI
"""

import os
import json
import time
import boto3
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Base class for LLM clients"""
    
    def __init__(self, model_id: str, max_tokens: int = 4096, temperature: float = 1.0, is_judge: bool = False):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.is_judge = is_judge
    
    @abstractmethod
    def invoke(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Invoke the LLM"""
        pass


class BedrockLLMClient(BaseLLMClient):
    """AWS Bedrock LLM Client"""
    
    def __init__(self, model_id: str, max_tokens: int = 4096, temperature: float = 1.0, 
                 is_judge: bool = False, region: str = None):
        super().__init__(model_id, max_tokens, temperature, is_judge)
        
        self.region = region or os.getenv('AWS_REGION', 'us-east-1')
        
        # AWS credentials
        aws_kwargs = {
            'service_name': 'bedrock-runtime',
            'region_name': self.region
        }
        
        aws_key = os.getenv('JUDGE_AWS_ACCESS_KEY_ID' if is_judge else 'AWS_ACCESS_KEY_ID')
        aws_secret = os.getenv('JUDGE_AWS_SECRET_ACCESS_KEY' if is_judge else 'AWS_SECRET_ACCESS_KEY')
        
        if aws_key and aws_secret:
            aws_kwargs['aws_access_key_id'] = aws_key
            aws_kwargs['aws_secret_access_key'] = aws_secret
        
        self.client = boto3.client(**aws_kwargs)
        self.provider = "aws"
    
    def invoke(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Invoke AWS Bedrock model"""
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{
                "role": "user",
                "content": prompt
            }]
        }
        
        if system_prompt:
            body["system"] = system_prompt
        
        start_time = time.time()
        
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )
            
            latency_ms = (time.time() - start_time) * 1000
            response_body = json.loads(response['body'].read())
            
            return {
                'success': True,
                'response': response_body['content'][0]['text'],
                'usage': response_body.get('usage', {}),
                'latency_ms': latency_ms,
                'provider': 'aws',
                'model_id': self.model_id,
                'input_tokens': response_body.get('usage', {}).get('input_tokens', 0),
                'output_tokens': response_body.get('usage', {}).get('output_tokens', 0)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'latency_ms': (time.time() - start_time) * 1000,
                'provider': 'aws',
                'model_id': self.model_id
            }


class OpenAILLMClient(BaseLLMClient):
    """OpenAI LLM Client"""
    
    def __init__(self, model_id: str, max_tokens: int = 4096, temperature: float = 1.0, 
                 is_judge: bool = False):
        super().__init__(model_id, max_tokens, temperature, is_judge)
        
        try:
            import openai
            self.openai = openai
            
            api_key = os.getenv('JUDGE_OPENAI_API_KEY' if is_judge else 'OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not found")
            
            self.client = openai.OpenAI(api_key=api_key)
            self.provider = "openai"
            
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
    
    def invoke(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Invoke OpenAI model"""
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        start_time = time.time()
        
        try:
            # Try with max_completion_tokens first (for newer models)
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_completion_tokens=self.max_tokens,
                    temperature=self.temperature
                )
            except Exception as e:
                # If max_completion_tokens fails, try max_tokens (for older models)
                if 'max_completion_tokens' in str(e) or 'max_tokens' in str(e):
                    response = self.client.chat.completions.create(
                        model=self.model_id,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                else:
                    raise e
            
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                'success': True,
                'response': response.choices[0].message.content,
                'usage': {
                    'input_tokens': response.usage.prompt_tokens,
                    'output_tokens': response.usage.completion_tokens
                },
                'latency_ms': latency_ms,
                'provider': 'openai',
                'model_id': self.model_id,
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'latency_ms': (time.time() - start_time) * 1000,
                'provider': 'openai',
                'model_id': self.model_id
            }


class LLMClientFactory:
    """Factory for creating LLM clients"""
    
    @staticmethod
    def create_client(provider: str, model_id: str, max_tokens: int = 4096, 
                     temperature: float = 1.0, is_judge: bool = False) -> BaseLLMClient:
        """Create appropriate LLM client based on provider"""
        
        provider = provider.lower()
        
        if provider == 'aws':
            return BedrockLLMClient(
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                is_judge=is_judge
            )
        elif provider == 'openai':
            return OpenAILLMClient(
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                is_judge=is_judge
            )
        else:
            raise ValueError(f"Unknown provider: {provider}. Supported: aws, openai")