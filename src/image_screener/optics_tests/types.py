"""Shared types for optics consistency tests."""

from typing import List, NamedTuple


class OpticsTestResult(NamedTuple):
    """Result from a single optics test."""

    score: float  # 0.0 (fails) to 1.0 (passes)
    violations: List[str]  # List of detected violations
    diagnostic_data: dict  # Additional data for visualization


class OpticsConsistencyResult(NamedTuple):
    """Complete optics consistency analysis result."""

    optics_score: float  # Overall score (0.0-1.0)
    frequency_test: OpticsTestResult
    edge_psf_test: OpticsTestResult
    dof_consistency_test: OpticsTestResult
    chromatic_aberration_test: OpticsTestResult
    noise_residual_test: OpticsTestResult  # Sensor noise residual test
    explanation: str  # Human-readable explanation

