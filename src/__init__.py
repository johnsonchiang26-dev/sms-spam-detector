"""SMS spam & phishing detection package.

Submodules are intentionally *not* eagerly imported here so that lightweight
consumers (e.g. ``src.utils`` in the test suite) can be imported without pulling
in heavy dependencies such as torch/transformers.
"""

__version__ = "0.1.0"

__all__ = ["utils", "dataset", "model", "train", "evaluate", "predict"]
