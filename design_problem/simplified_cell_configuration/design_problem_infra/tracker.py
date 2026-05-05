"""Design step tracking for battery pack optimization.

This module provides classes and functions to track the design process,
including design steps, validation results, and CRDAL feedback.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from system_infra.model_adapters import ModelResponse


@dataclass
class DesignStep:
    """Represents a single step in the battery pack design process.

    Attributes:
        step_number: Current step count (0 for initial design)
        raw_response: Raw text output from the LLM
        parsing_success: Whether parsing was successful
        design_dict: Parsed design specifications (cell_locations, dimensions, voltage, etc.)
        is_valid: Whether this step passed validation
        validation_results: Detailed validation results (if validation performed)
        crdal_feedback: Feedback from CRDAL agent (if applicable)
        timestamp: When this step was created
        metadata: Additional metadata (model used, design_agent_total_tokens_used, crdal_agent_total_tokens_used, etc.)
    """

    step_number: int
    text_response: str
    parsing_success: bool = False
    design_dict: Optional[Dict[str, Any]] = None
    is_valid: bool = False
    validation_results: Optional[Dict[str, Any]] = None
    crdal_feedback: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default=None):
        """Dict-like get method for backward compatibility.

        Args:
            key: Key to retrieve
            default: Default value if key not found

        Returns:
            Value for key, or default if not found
        """
        if key == "raw_response":
            return self.text_response
        elif key == "step_number":
            return self.step_number
        elif key == "parsing_success":
            return self.parsing_success
        elif key == "is_valid":
            return self.is_valid
        elif self.design_dict and key in self.design_dict:
            return self.design_dict[key]
        else:
            return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the design step
        """
        d = self.design_dict or {}
        return {
            "step_number": self.step_number,
            "text_response": self.text_response,

            "cell_locations": d.get("cell_locations"),
            "cell_spacing": d.get("cell_spacing"),

            "num_cells_width": d.get("num_cells_width"),
            "num_cells_depth": d.get("num_cells_depth"),
            "num_cells_height": d.get("num_cells_height"),
            "design_width": d.get("design_width"),
            "design_depth": d.get("design_depth"),
            "design_height": d.get("design_height"),

            "cell_count": d.get("cell_count"),
            "surface_area": d.get("surface_area"),
            "design_volume": d.get("design_volume"),
            "max_temperature_C": d.get("max_temperature_C"),

            "series_count": d.get("series_count"),
            "parallel_count": d.get("parallel_count"),

            "design_voltage": d.get("design_voltage"),
            "design_capacity": d.get("design_capacity"),
            "max_current": d.get("max_current"),

            "is_valid": self.is_valid,
            "validation_results": self.validation_results,
            "parsing_success": self.parsing_success,
            "crdal_feedback": self.crdal_feedback,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Generate a human-readable summary of this step.

        Returns:
            Formatted summary string
        """
        lines = [
            f"=== Design Step {self.step_number} ===",
            f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Parsing Success: {self.parsing_success}",
            f"Valid: {self.is_valid}",
        ]

        if self.design_dict:
            lines.append("\nDesign Features:")
            for key, value in self.design_dict.items():
                if value is not None:
                    lines.append(f"  {key}: {value}")

        if self.validation_results:
            lines.append("\nValidation Results:")
            for key, value in self.validation_results.items():
                if value is not None:
                    lines.append(f"  {key}: {value}")

        if self.crdal_feedback:
            lines.append(f"\nCRDAL Feedback: {self.crdal_feedback[:100]}...")

        return "\n".join(lines)


@dataclass
class DesignTrace:
    """Tracks the complete design optimization trace.

    Attributes:
        problem_query: Original design problem/query
        steps: List of design steps in chronological order
        required_specs: Required specifications for validation
        final_design: Final accepted design (if process completed successfully)
        metadata: Additional metadata about the trace
    """

    problem_query: str
    steps: List[DesignStep] = field(default_factory=list)
    required_specs: Optional[Dict[str, Any]] = None
    final_design: Optional[DesignStep] = None
    messages_history: List = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: DesignStep):
        """Add a design step to the trace.

        Args:
            step: DesignStep to add
        """
        self.steps.append(step)

        # Update final design if this step is valid
        if step.is_valid:
            self.final_design = step

    def add_message_history(self, message_history):
        """Add message_history to the trace.

        Args:
            message_history to add
        """
        self.messages_history.append(message_history)

    def get_current_step(self) -> Optional[DesignStep]:
        """Get the most recent design step.

        Returns:
            Most recent DesignStep, or None if no steps
        """
        return self.steps[-1] if self.steps else None

    def get_step_count(self) -> int:
        """Get the total number of steps.

        Returns:
            Number of steps in the trace
        """
        return len(self.steps)

    def is_complete(self) -> bool:
        """Check if the design process is complete.

        Returns:
            True if we have a valid final design, False otherwise
        """
        return self.final_design is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the trace
        """
        def _sum(key):
            return sum(s.metadata.get(key) or 0 for s in self.steps)

        design_agent_input_tokens_sum       = _sum("design_agent_input_tokens")
        design_agent_output_tokens_sum      = _sum("design_agent_output_tokens")
        design_agent_thinking_tokens_sum    = _sum("design_agent_thinking_tokens")
        design_agent_total_tokens_used_sum  = _sum("design_agent_total_tokens_used")
        crdal_agent_input_tokens_sum        = _sum("crdal_agent_input_tokens")
        crdal_agent_output_tokens_sum       = _sum("crdal_agent_output_tokens")
        crdal_agent_thinking_tokens_sum     = _sum("crdal_agent_thinking_tokens")
        crdal_agent_total_tokens_used_sum   = _sum("crdal_agent_total_tokens_used")

        return {
            "problem_query": self.problem_query,
            "steps": [step.to_dict() for step in self.steps],
            "required_specs": self.required_specs,
            "final_design": self.final_design.to_dict() if self.final_design else None,
            "metadata": self.metadata,
            "total_steps": len(self.steps),
            "is_complete": self.is_complete(),
            "design_agent_input_tokens_sum":        design_agent_input_tokens_sum,
            "design_agent_output_tokens_sum":       design_agent_output_tokens_sum,
            "design_agent_thinking_tokens_sum":     design_agent_thinking_tokens_sum,
            "design_agent_total_tokens_used_sum":   design_agent_total_tokens_used_sum,
            "crdal_agent_input_tokens_sum":         crdal_agent_input_tokens_sum,
            "crdal_agent_output_tokens_sum":        crdal_agent_output_tokens_sum,
            "crdal_agent_thinking_tokens_sum":      crdal_agent_thinking_tokens_sum,
            "crdal_agent_total_tokens_used_sum":    crdal_agent_total_tokens_used_sum,
            "total_tokens_used_sum": design_agent_total_tokens_used_sum + crdal_agent_total_tokens_used_sum,
            "messages_history": self.messages_history,
        }

    def save_to_json(self, filepath: Optional[str] = None, indent: int = 2) -> str:
        """Convert to JSON and optionally save to file.

        Args:
            filepath: Optional path to save the JSON file. If None, only returns JSON string.
            indent: Number of spaces for indentation (default: 2)

        Returns:
            JSON string representation of the trace
        """
        import re

        # Dump with full indentation first, then collapse cell_locations to a single line
        json_str = json.dumps(self.to_dict(), indent=indent, default=str)

        # Step 1: compact individual [x, y, z] coordinate triplets onto one line each
        json_str = re.sub(
            r'\[\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+)\s*\]',
            r'[\1, \2, \3]',
            json_str
        )

        # Step 2: collapse the entire cell_locations array onto a single line
        def _collapse_cell_locations(m):
            return re.sub(r'\s+', ' ', m.group(0))

        json_str = re.sub(
            r'"cell_locations":\s*\[(?:\s*\[-?[\d.]+,\s*-?[\d.]+,\s*-?[\d.]+\],?)*\s*\]',
            _collapse_cell_locations,
            json_str
        )

        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)

        return json_str

    def summary(self) -> str:
        """Generate a human-readable summary of the trace.

        Returns:
            Formatted summary string
        """
        lines = [
            "=" * 60,
            "DESIGN TRACE SUMMARY",
            "=" * 60,
            f"Problem: {self.problem_query}",
            f"Total Steps: {len(self.steps)}",
            f"Complete: {self.is_complete()}",
        ]

        if self.required_specs:
            lines.append("\nRequired Specifications:")
            for key, value in self.required_specs.items():
                lines.append(f"  {key}: {value}")

        if self.final_design:
            lines.append("\nFinal Design:")
            lines.append(f"  Step Number: {self.final_design.step_number}")
            if self.final_design.design_dict:
                for key, value in self.final_design.design_dict.items():
                    if value is not None:
                        lines.append(f"  {key}: {value}")

        return "\n".join(lines)


def step_conversion(
        step_number: int,
        model_response: ModelResponse,
        design_dict: Optional[Dict[str, Any]] = None,
        validation_results: Optional[Dict[str, Any]] = None,
        crdal_feedback: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
        ) -> DesignStep:
    """Convert a ModelResponse to a DesignStep.

    Args:
        step_number: Current step count
        model_response: ModelResponse object from the model adapter
        design_dict: Parsed design specifications (cell_locations, dimensions, voltage, etc.)
        validation_results: Detailed validation results (if validation performed)
        crdal_feedback: Feedback from CRDAL agent (if applicable)
        metadata: Optional additional metadata

    Returns:
        DesignStep object with parsed information
    """

    # Handle error in response
    if model_response.error:
        print(f"[ERROR] Model response has error: {model_response.error}")
        return DesignStep(
            step_number=step_number,
            text_response="",
            parsing_success=False,
            is_valid=False,
            metadata={"error": model_response.error}
        )

    # Extract text from response
    raw_text = model_response.text

    # Determine parsing success based on whether design_dict is valid
    parsing_success = design_dict is not None

    # If validation_results contains extraction_error, parsing failed
    if validation_results and "extraction_error" in validation_results:
        parsing_success = False

    # Determine overall validity from validation_results
    is_valid = False
    if validation_results:
        # Check if all validation results passed (excluding extraction_error)
        validatable = [
            result for key, result in validation_results.items()
            if hasattr(result, 'is_valid') and key != "extraction_error"
        ]
        is_valid = len(validatable) > 0 and all(r.is_valid for r in validatable)

    # Merge metadata from model response and additional metadata
    step_metadata = {}
    if model_response.metadata:
        step_metadata.update(model_response.metadata)
    if metadata:
        step_metadata.update(metadata)

    # Add "design_agent_" prefix to design agent token fields
    _token_renames = {
        "tokens_used":     "design_agent_total_tokens_used",
        "input_tokens":    "design_agent_input_tokens",
        "output_tokens":   "design_agent_output_tokens",
        "thinking_tokens": "design_agent_thinking_tokens",
    }
    for old_key, new_key in _token_renames.items():
        if old_key in step_metadata:
            step_metadata[new_key] = step_metadata.pop(old_key)

    # Create and return DesignStep
    step = DesignStep(
        step_number=step_number,
        text_response=raw_text,
        parsing_success=parsing_success,
        design_dict=design_dict,
        is_valid=is_valid,
        validation_results=validation_results,
        crdal_feedback=crdal_feedback,
        metadata=step_metadata
    )

    return step


def record_crdal_response(step: DesignStep, crdal_response: ModelResponse) -> None:
    """Record CRDAL agent response text and token usage into an existing step.

    Args:
        step: DesignStep to update (in-place)
        crdal_response: ModelResponse from the CRDAL agent
    """
    step.crdal_feedback = crdal_response.text
    if crdal_response.metadata:
        meta = crdal_response.metadata
        step.metadata["crdal_agent_total_tokens_used"]  = meta.get("tokens_used")
        step.metadata["crdal_agent_input_tokens"]       = meta.get("input_tokens")
        step.metadata["crdal_agent_output_tokens"]      = meta.get("output_tokens")
        step.metadata["crdal_agent_thinking_tokens"]    = meta.get("thinking_tokens")
