"""Compatibility entry point. Actual method is GradientExplainer expected gradients."""
from pathlib import Path
import runpy
if __name__ == "__main__":
 runpy.run_path(str(Path(__file__).with_name("attribute_expected_gradients.py")),run_name="__main__")
