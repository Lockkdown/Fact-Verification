"""
LLM Client - Unified interface cho multiple LLM providers.

Supports:
- OpenAI (GPT-4o-mini)
- DeepSeek
- Google Gemini
- Together AI (Llama)
- Zhipu AI (GLM)

Author: Lockdown
Date: Nov 10, 2025
"""

import time
import logging
import asyncio
import sys
from typing import Dict, Any, Optional
import requests
import aiohttp

# WINDOWS FIX: Set proper event loop policy to avoid SSL errors
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified LLM client cho multiple providers.
    Handles API calls, retries, và error handling.
    """
    
    def __init__(self, retry_config: Optional[Dict[str, Any]] = None):
        """
        Args:
            retry_config: {max_attempts: int, backoff_seconds: float}
        """
        self.retry_config = retry_config or {"max_attempts": 3, "backoff_seconds": 2}
        self._session: Optional[aiohttp.ClientSession] = None
        
    def generate(
        self,
        model: str,
        prompt: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.7,
        max_tokens: int = 800,
        **kwargs
    ) -> str:
        """
        Generate response from LLM.
        
        Args:
            model: Model name
            prompt: Input prompt
            api_key: API key
            base_url: Base URL for API
            temperature: Sampling temperature
            max_tokens: Max output tokens
            
        Returns:
            Generated text
        """
        
        # Detect provider from base_url
        if "openai" in base_url:
            return self._call_openai(model, prompt, api_key, base_url, temperature, max_tokens)
        elif "deepseek" in base_url:
            return self._call_deepseek(model, prompt, api_key, base_url, temperature, max_tokens)
        elif "generativelanguage" in base_url or "googleapis" in base_url:
            return self._call_gemini(model, prompt, api_key, base_url, temperature, max_tokens)
        elif "groq" in base_url:
            return self._call_groq(model, prompt, api_key, base_url, temperature, max_tokens)
        elif "mistral" in base_url:
            return self._call_mistral(model, prompt, api_key, base_url, temperature, max_tokens)
        elif "openrouter" in base_url:
            return self._call_openrouter(model, prompt, api_key, base_url, temperature, max_tokens)
        elif "newapi" in base_url or "douplett" in base_url:
            # NewAPI uses OpenAI-compatible format
            return self._call_openai(model, prompt, api_key, base_url, temperature, max_tokens)
        elif "together" in base_url:
            return self._call_together(model, prompt, api_key, base_url, temperature, max_tokens)
        elif "bigmodel" in base_url:
            return self._call_zhipu(model, prompt, api_key, base_url, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown provider from base_url: {base_url}")
    
    def _retry_wrapper(self, func, *args, **kwargs):
        """Retry wrapper với exponential backoff."""
        
        max_attempts = self.retry_config['max_attempts']
        backoff = self.retry_config['backoff_seconds']
        
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise
                
                wait_time = backoff * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
    
    def _call_openai(
        self,
        model: str,
        prompt: str,
        api_key: str,
        base_url: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Call OpenAI API (GPT-4o-mini)."""
        
        def _call():
            url = f"{base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            logger.debug(f"Calling {url} with model={model}")
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            # Log error details if request fails
            if not response.ok:
                logger.error(f"API Error {response.status_code}: {response.text}")
            
            response.raise_for_status()
            
            data = response.json()
            return data['choices'][0]['message']['content']
        
        return self._retry_wrapper(_call)
    
    def _call_deepseek(
        self,
        model: str,
        prompt: str,
        api_key: str,
        base_url: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Call DeepSeek API."""
        
        def _call():
            url = f"{base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return data['choices'][0]['message']['content']
        
        return self._retry_wrapper(_call)
    
    def _call_gemini(
        self,
        model: str,
        prompt: str,
        api_key: str,
        base_url: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Call Google Gemini API."""
        
        def _call():
            # Gemini API uses different format
            url = f"{base_url}/models/{model}:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens
                }
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Debug logging
            if 'candidates' not in data:
                logger.error(f"Gemini response missing 'candidates': {data}")
                raise ValueError(f"Invalid Gemini response format: {data}")
            
            # Handle blocked or empty responses
            if not data['candidates']:
                logger.warning("Gemini returned empty candidates")
                raise ValueError("Gemini returned no candidates (possibly blocked content)")
            
            candidate = data['candidates'][0]
            
            # Check for safety blocks
            if 'content' not in candidate:
                finish_reason = candidate.get('finishReason', 'UNKNOWN')
                safety_ratings = candidate.get('safetyRatings', [])
                logger.warning(f"Gemini blocked response. Reason: {finish_reason}, Safety: {safety_ratings}")
                raise ValueError(f"Gemini blocked response: {finish_reason}")
            
            return candidate['content']['parts'][0]['text']
        
        return self._retry_wrapper(_call)
    
    def _call_together(
        self,
        model: str,
        prompt: str,
        api_key: str,
        base_url: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Call Together AI API (Llama)."""
        
        def _call():
            url = f"{base_url}/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return data['choices'][0]['message']['content']
        
        return self._retry_wrapper(_call)
    
    def _call_zhipu(
        self,
        model: str,
        prompt: str,
        api_key: str,
        base_url: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Call Zhipu AI API (GLM)."""
        
        def _call():
            url = f"{base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return data['choices'][0]['message']['content']
        
        return self._retry_wrapper(_call)
    
    def _call_groq(
        self,
        model: str,
        prompt: str,
        api_key: str,
        base_url: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Call Groq API (Llama via Groq)."""
        
        def _call():
            url = f"{base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return data['choices'][0]['message']['content']
        
        return self._retry_wrapper(_call)
    
    def _call_mistral(
        self,
        model: str,
        prompt: str,
        api_key: str,
        base_url: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Call Mistral AI API."""
        
        def _call():
            url = f"{base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return data['choices'][0]['message']['content']
        
        return self._retry_wrapper(_call)
    
    def _call_openrouter(
        self,
        model: str,
        prompt: str,
        api_key: str,
        base_url: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Call OpenRouter API (unified access to many models)."""
        
        def _call():
            url = f"{base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Disable reasoning for models that support it (to get direct responses)
            # GPT-5 and Grok 4 have reasoning modes that can cause verbose/empty responses
            if "gpt-5" in model.lower() or "grok-4" in model.lower():
                payload["reasoning"] = {"enabled": False}
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return data['choices'][0]['message']['content']
        
        return self._retry_wrapper(_call)
    
    # ========== ASYNC METHODS FOR CONCURRENT API CALLS ==========
    
    async def generate_async(
        self,
        model: str,
        prompt: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.7,
        max_tokens: int = 800,
        **kwargs
    ) -> str:
        """
        Async version of generate() for concurrent API calls.
        
        Args:
            Same as generate()
            
        Returns:
            Generated text
        """
        
        # Ensure fresh session for each call to avoid event loop closed errors
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        
        # Most providers use OpenAI-compatible format
        return await self._call_openai_async(model, prompt, api_key, base_url, temperature, max_tokens)
    
    async def close_session(self):
        """Close aiohttp session if open"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Close aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            # WINDOWS FIX: Sleep to allow SSL connections to close cleanly
            # This prevents "Fatal error on SSL transport" and "WinError 10038"
            await asyncio.sleep(0.250)
    
    async def _retry_wrapper_async(self, coro, *args, **kwargs):
        """Async retry wrapper with exponential backoff."""
        
        max_attempts = 5  # Increased from 3 for better rate limit handling
        backoff = 4       # Increased from 2
        
        for attempt in range(max_attempts):
            try:
                return await coro(*args, **kwargs)
            except Exception as e:
                is_rate_limit = "429" in str(e) or "Too Many Requests" in str(e)
                
                if attempt == max_attempts - 1:
                    logger.error(f"❌ All {max_attempts} attempts failed. Last error: {e}")
                    raise
                
                # Calculate wait time
                wait_time = backoff * (2 ** attempt)
                
                # If rate limited, add extra jitter/padding
                if is_rate_limit:
                    wait_time += 5 
                    logger.warning(f"⚠️ Rate Limited (429). Waiting {wait_time}s before retry {attempt + 2}/{max_attempts}...")
                else:
                    logger.warning(f"⚠️ Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                
                await asyncio.sleep(wait_time)
    
    async def _call_openai_async(
        self,
        model: str,
        prompt: str,
        api_key: str,
        base_url: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Async call to OpenAI/OpenRouter API (and other OpenAI-compatible APIs)."""
        
        async def _call():
            url = f"{base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Disable reasoning for models that support it (to get direct responses)
            # GPT-5 and Grok 4 have reasoning modes that can cause verbose/empty responses
            if "gpt-5" in model.lower() or "grok-4" in model.lower():
                payload["reasoning"] = {"enabled": False}
            
            logger.debug(f"Async calling {url} with model={model}")
            
            session = await self._get_session()
            async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if not response.ok:
                    error_text = await response.text()
                    logger.error(f"API Error {response.status}: {error_text}")
                    response.raise_for_status()
                
                data = await response.json()
                content = data['choices'][0]['message']['content']
                
                # Check for empty response - treat as error to trigger retry
                if not content or not content.strip():
                    finish_reason = data['choices'][0].get('finish_reason', 'unknown')
                    logger.warning(f"Empty response from {model}, finish_reason={finish_reason}")
                    raise ValueError(f"Empty response from {model} (finish_reason={finish_reason})")
                
                return content
        
        return await self._retry_wrapper_async(_call)
