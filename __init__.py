"""
MomentEmu: A lightweight, interpretable polynomial emulator for smooth mappings.

This package implements the moment-projection polynomial emulator introduced in 
Zhang (2025), building interpretable, closed-form polynomial emulators via moment 
matrices with millisecond-level inference and symbolic transparency.
"""

from .MomentEmu import PolyEmu, evaluate_emulator, symbolic_polynomial_expressions

__version__ = "1.0.0"
__author__ = "Your Name"
__all__ = ["PolyEmu", "evaluate_emulator", "symbolic_polynomial_expressions"]