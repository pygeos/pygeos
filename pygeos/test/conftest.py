import sys
import os

# Small hack to make "common.py" importable without using relative
# imports. Pytest is tricky with relative imports.
sys.path.append(os.path.join(os.path.dirname(__file__)))
