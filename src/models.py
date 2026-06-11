"""Pydantic models for function definitions and function calls."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator


def type_matches(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "number":
        return isinstance(value, (int, float))
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "null":
        return value is None
    return isinstance(value, (str, int, float, bool, dict, list))


class ParameterDefinition(BaseModel):
    """Definition of a single parameter for a function."""

    type: str = Field(
        description=(
            "Type of the parameter "
            "(string, number, boolean, etc.)"
        ),
    )


class ReturnDefinition(BaseModel):
    """Definition of the return type for a function."""

    type: str = Field(description="Return type of the function")


class FunctionDefinition(BaseModel):
    """Schema for a function definition from the input file."""

    name: str = Field(description="Name of the function")
    description: str = Field(
        description="Description of what the function does",
    )
    parameters: Dict[str, ParameterDefinition] = Field(
        default_factory=dict,
        description="Dictionary of parameter names to their definitions",
    )
    returns: Optional[ReturnDefinition] = Field(
        default=None, description="Return type definition",
    )


class PromptEntry(BaseModel):
    """A single prompt from the input tests file."""

    prompt: str = Field(description="The natural language prompt")


class FunctionCallOutput(BaseModel):
    """The LLM-generated function call JSON before wrapping."""

    function: str = Field(description="The function name selected by the LLM")
    arguments: Dict[str, Any] = Field(
        description="The arguments extracted for the function",
    )

    @model_validator(mode="after")
    def validate_function_name(self) -> FunctionCallOutput:
        """Validate that the function name is not empty."""
        if not self.function:
            raise ValueError("Function name cannot be empty")
        return self
