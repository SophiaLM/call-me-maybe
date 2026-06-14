from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel


class FunctionParameter(BaseModel):
    """A single parameter of a function."""

    type: str


class FunctionDefinition(BaseModel):
    """Definition of a callable function with its parameters."""

    name: str
    description: str
    parameters: Dict[str, FunctionParameter]
    returns: Dict[str, str]


class FunctionCall(BaseModel):
    """Result of processing a prompt: chosen function and args."""

    prompt: str
    fn_name: str
    args: Dict[str, Any]
