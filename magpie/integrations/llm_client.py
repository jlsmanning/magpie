"""
Claude API client for Magpie.

Wrapper around Anthropic's Claude API for LLM interactions.
"""

import typing
import json
import anthropic
from magpie.utils.config import Config


class ClaudeClient:
    """
    Client for interacting with Claude API.
    
    Provides simple interface for chat completions and structured outputs.
    Handles API calls, error handling, and response parsing.
    """
    
    def __init__(
        self,
        api_key: typing.Optional[str] = None,
        model: typing.Optional[str] = None
    ):
        """
        Initialize Claude client.
        
        Args:
            api_key: Anthropic API key. If None, uses Config.ANTHROPIC_API_KEY
            model: Model name. If None, uses Config.LLM_MODEL
        """
        self.api_key = api_key or Config.ANTHROPIC_API_KEY
        self.model = model or Config.LLM_MODEL
        
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable."
            )
        
        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Track usage statistics
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    def chat(
        self,
        messages: typing.List[typing.Dict[str, str]],
        system: typing.Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 1.0
    ) -> str:
        """
        Send chat message to Claude and get text response.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
                      Example: [{"role": "user", "content": "Hello!"}]
            system: Optional system prompt to guide Claude's behavior
            max_tokens: Maximum tokens in response (default: 1024)
            temperature: Randomness/creativity (0.0-1.0, default: 1.0)
                        Lower = more deterministic, Higher = more creative
        
        Returns:
            Claude's text response
            
        Example:
            >>> client = ClaudeClient()
            >>> response = client.chat(
            ...     messages=[{"role": "user", "content": "What is RAG?"}],
            ...     system="You are a helpful AI assistant."
            ... )
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system if system else anthropic.NOT_GIVEN,
                messages=messages
            )
            
            # Track token usage
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            
            # Extract text from response
            # Claude returns content as list of content blocks
            text_blocks = [
                block.text for block in response.content
                if hasattr(block, 'text')
            ]
            
            return "\n".join(text_blocks)
            
        except anthropic.APIError as e:
            raise RuntimeError(f"Claude API error: {e}")
    
    def chat_with_json(
        self,
        messages: typing.List[typing.Dict[str, str]],
        system: typing.Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0
    ) -> typing.Dict[str, typing.Any]:
        """
        Send chat message expecting JSON response.
        
        Useful for structured outputs like query decomposition or
        intent parsing. Uses lower temperature for more deterministic
        structured output.
        
        Args:
            messages: Message history
            system: System prompt (should instruct to output JSON)
            max_tokens: Max tokens (default: 2048 for structured output)
            temperature: Temperature (default: 0.0 for deterministic)
            
        Returns:
            Parsed JSON dict from Claude's response
            
        Raises:
            ValueError: If response is not valid JSON
            
        Example:
            >>> response = client.chat_with_json(
            ...     messages=[{"role": "user", "content": "Parse this: ..."}],
            ...     system="Return only valid JSON."
            ... )
        """
        # Add JSON instruction to system prompt if not already there
        if system and "json" not in system.lower():
            system = f"{system}\n\nReturn your response as valid JSON only, with no other text."
        elif not system:
            system = "Return your response as valid JSON only, with no other text."
        
        # Get response
        response_text = self.chat(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Parse JSON
        try:
            # Remove markdown code fences if present (```json ... ```)
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                # Find JSON between code fences
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1])  # Remove first and last line
            
            return json.loads(cleaned)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from Claude response: {e}\nResponse: {response_text}")
    
    def get_usage_stats(self) -> typing.Dict[str, int]:
        """
        Get token usage statistics.
        
        Returns:
            Dict with input_tokens, output_tokens, total_tokens
        """
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens
        }
    
    def estimate_cost(self) -> float:
        """
        Estimate API cost based on token usage.
        
        Uses current pricing for Claude Sonnet 4.5:
        - Input: $3 per million tokens
        - Output: $15 per million tokens
        
        Returns:
            Estimated cost in USD
            
        Note: Pricing may change. Check Anthropic's website for current rates.
        """
        # Pricing as of Jan 2025 for Claude Sonnet 4.5
        input_cost_per_million = 3.0
        output_cost_per_million = 15.0
        
        input_cost = (self.total_input_tokens / 1_000_000) * input_cost_per_million
        output_cost = (self.total_output_tokens / 1_000_000) * output_cost_per_million
        
        return input_cost + output_cost
