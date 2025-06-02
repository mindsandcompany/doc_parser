import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

_log = logging.getLogger(__name__)

class PromptManager:
    """프롬프트 및 API 설정 관리 클래스"""
    
    def __init__(self, prompts_file: Optional[str] = None):
        """
        Args:
            prompts_file: 프롬프트 JSON 파일 경로 (기본값: prompts.json)
        """
        if prompts_file is None:
            self.prompts_file = Path(__file__).parent / "prompts.json"
        else:
            self.prompts_file = Path(prompts_file)
        
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """설정 JSON 파일 로드"""
        try:
            with open(self.prompts_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            _log.error(f"설정 파일을 찾을 수 없습니다: {self.prompts_file}")
            return {}
        except json.JSONDecodeError as e:
            _log.error(f"설정 파일 JSON 파싱 오류: {e}")
            return {}
    
    def get_api_config(self, provider: str) -> Optional[Dict[str, Any]]:
        """
        API 설정 반환
        
        Args:
            provider: API 제공업체 (예: openrouter, openai)
        
        Returns:
            API 설정 딕셔너리 또는 None
        """
        try:
            api_config = self._config["api_config"][provider].copy()
            
            # 환경변수에서 API 키 우선 조회
            env_key_name = f"{provider.upper()}_API_KEY"
            env_api_key = os.getenv(env_key_name)
            if env_api_key:
                api_config["api_key"] = env_api_key
                _log.info(f"환경변수에서 {provider} API 키 로드됨")
            
            return api_config
        except KeyError:
            _log.error(f"API 설정을 찾을 수 없습니다: {provider}")
            return None
    
    def get_prompt_config(self, category: str, prompt_type: str) -> Optional[Dict[str, Any]]:
        """
        특정 프롬프트 설정 반환
        
        Args:
            category: 프롬프트 카테고리 (예: metadata_extraction, summary)
            prompt_type: 프롬프트 타입 (예: korean_financial, english_general)
        
        Returns:
            프롬프트 설정 딕셔너리 또는 None
        """
        try:
            return self._config[category][prompt_type]
        except KeyError:
            _log.error(f"프롬프트를 찾을 수 없습니다: {category}.{prompt_type}")
            return None
    
    def get_system_prompt(self, category: str, prompt_type: str) -> Optional[str]:
        """시스템 프롬프트 반환"""
        config = self.get_prompt_config(category, prompt_type)
        return config.get("system") if config else None
    
    def get_user_prompt_template(self, category: str, prompt_type: str) -> Optional[str]:
        """사용자 프롬프트 템플릿 반환"""
        config = self.get_prompt_config(category, prompt_type)
        return config.get("user") if config else None
    
    def format_user_prompt(self, category: str, prompt_type: str, **kwargs) -> Optional[str]:
        """
        사용자 프롬프트 템플릿에 변수를 삽입하여 완성된 프롬프트 반환
        
        Args:
            category: 프롬프트 카테고리
            prompt_type: 프롬프트 타입
            **kwargs: 템플릿에 삽입할 변수들
        
        Returns:
            완성된 프롬프트 문자열 또는 None
        """
        template = self.get_user_prompt_template(category, prompt_type)
        if template is None:
            return None
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            _log.error(f"프롬프트 템플릿 변수 오류: {e}")
            return None
    
    def get_messages(self, category: str, prompt_type: str, **kwargs) -> Optional[List[Dict[str, Any]]]:
        """
        OpenAI API용 메시지 형식으로 반환
        
        Args:
            category: 프롬프트 카테고리
            prompt_type: 프롬프트 타입
            **kwargs: 템플릿에 삽입할 변수들
        
        Returns:
            OpenAI API 메시지 리스트 또는 None
        """
        system_prompt = self.get_system_prompt(category, prompt_type)
        user_prompt = self.format_user_prompt(category, prompt_type, **kwargs)
        
        if system_prompt is None or user_prompt is None:
            return None
        
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }
        ]
    
    def get_model_config(self, category: str, prompt_type: str) -> Dict[str, Any]:
        """
        모델 설정 반환 (model, seed, api_provider 등)
        
        Returns:
            모델 설정 딕셔너리
        """
        config = self.get_prompt_config(category, prompt_type)
        if config is None:
            return {
                "model": "google/gemma-3-12b-it:free", 
                "seed": 3,
                "api_provider": "openrouter"
            }
        
        return {
            "model": config.get("model", "google/gemma-3-12b-it:free"),
            "seed": config.get("seed", 3),
            "api_provider": config.get("api_provider", "openrouter")
        }
    
    def get_complete_config(self, category: str, prompt_type: str) -> Optional[Dict[str, Any]]:
        """
        프롬프트 설정 + API 설정을 모두 포함한 완전한 설정 반환
        
        Returns:
            완전한 설정 딕셔너리 또는 None
        """
        prompt_config = self.get_prompt_config(category, prompt_type)
        if prompt_config is None:
            return None
        
        api_provider = prompt_config.get("api_provider", "openrouter")
        api_config = self.get_api_config(api_provider)
        
        if api_config is None:
            return None
        
        return {
            "prompt": prompt_config,
            "api": api_config
        }
    
    def list_available_prompts(self) -> Dict[str, List[str]]:
        """사용 가능한 모든 프롬프트 목록 반환"""
        result = {}
        for category, prompts in self._config.items():
            if category != "api_config":  # API 설정 제외
                result[category] = list(prompts.keys())
        return result
    
    def list_api_providers(self) -> List[str]:
        """사용 가능한 API 제공업체 목록 반환"""
        return list(self._config.get("api_config", {}).keys())
    
    def get_prompt_description(self, category: str, prompt_type: str) -> Optional[str]:
        """프롬프트 설명 반환"""
        config = self.get_prompt_config(category, prompt_type)
        return config.get("description") if config else None