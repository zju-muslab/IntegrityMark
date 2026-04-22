"""
Evaluation scripts for IntegrityMark watermarking system.

This module contains comprehensive evaluation scripts for different scenarios:
- eval.py: Main evaluation script
- eval_cross_source_attack.py: Cross-source attack robustness
- eval_in_source_attack.py: In-source attack robustness
- eval_distortion.py: Distortion robustness
- eval_cross_domain.py: Cross-domain generalization
- eval_language.py: Language transfer
- eval_detection_speed.py: Detection latency benchmark
"""

import sys
import os

# Add parent directory to path so scripts can import from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

__all__ = [
    'eval',
    'eval_cross_source_attack',
    'eval_in_source_attack',
    'eval_distortion',
    'eval_cross_domain',
    'eval_language',
    'eval_detection_speed',
]
