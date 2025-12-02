"""
Simple test script to verify the environment is set up correctly
Run this before running transformer.py to check if everything works
"""

import sys

print("Testing Python environment...")
print(f"Python version: {sys.version}")

# Test basic imports
try:
    import torch
    print(f"OK: PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"ERROR: PyTorch import failed: {e}")
    sys.exit(1)

try:
    import sentencepiece as spm
    print("OK: SentencePiece imported successfully")
except ImportError as e:
    print(f"ERROR: SentencePiece import failed: {e}")
    print("   Install with: pip install sentencepiece")
    sys.exit(1)

try:
    import numpy as np
    print(f"OK: NumPy version: {np.__version__}")
except ImportError as e:
    print(f"ERROR: NumPy import failed: {e}")
    sys.exit(1)

# Test basic operations
print("\nTesting basic operations...")
try:
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5, 6])
    z = x + y
    print("OK: Tensor operations work")
except Exception as e:
    print(f"ERROR: Tensor operations failed: {e}")
    sys.exit(1)

print("\nAll tests passed! Your environment is ready!")
print("You can now run: python transformer.py")

