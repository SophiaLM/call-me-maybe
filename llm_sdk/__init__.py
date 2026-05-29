"""llm_sdk - LLM SDK wrapper for local model inference.

This package re-exports Small_LLM_Model from the inner llm_sdk package
for backward compatibility with the call_me_maybe project.
"""

from llm_sdk.llm_sdk import Small_LLM_Model

__all__ = ["Small_LLM_Model"]
