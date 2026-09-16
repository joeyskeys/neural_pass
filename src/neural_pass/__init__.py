"""neural_pass — neural post-render pass after Arnold (scaffold).

Vision: Arnold condition AOVs + user mask → generative style model →
stylized drawing. Style configs drive both generation and post-training.
Accepted finals feed a labeled flywheel for iterative fine-tuning.
"""

__version__ = "0.1.0"
