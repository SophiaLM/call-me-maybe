"""Custom exceptions for the call_me_maybe project."""


class CallMeMaybeError(Exception):
    """Base exception for all project errors."""


class VocabularyError(CallMeMaybeError):
    """Raised when vocabulary loading or lookup fails."""


class DecoderError(CallMeMaybeError):
    """Raised when constrained decoding encounters an issue."""


class LoaderError(CallMeMaybeError):
    """Raised when input file loading or parsing fails."""


class GeneratorError(CallMeMaybeError):
    """Raised when LLM generation fails."""


class ValidationError(CallMeMaybeError):
    """Raised when output validation fails."""
