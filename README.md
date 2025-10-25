[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/ZF4v4FfT)
# MiniTorch Module 2

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

**Tensors** - Extending Autodifferentiation to Multidimensional Arrays

* Docs: https://minitorch.github.io/
* Overview: https://minitorch.github.io/module2/module2/

## Overview

Module 2 introduces **Tensors** - multidimensional arrays that extend the scalar autodifferentiation system from Module 1. While the scalar system is correct, it's inefficient due to Python overhead. Tensors solve this by grouping operations together and enabling faster implementations.

## Installation

See [installation.md](installation.md) for detailed setup instructions.

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev,extra]"

# Sync files from Module 1
python sync_previous_module.py ../Module-1 .

# Verify installation
python -c "import minitorch; print('Success!')"

# Run tests
pytest -m task2_1  # Tensor data and indexing
pytest -m task2_2  # Tensor broadcasting
pytest -m task2_3  # Tensor operations
pytest -m task2_4  # Tensor autodifferentiation

# Train tensor-based model
python project/run_tensor.py
```

## Tasks

### Task 2.1: Tensor Data - Indexing
**File to Edit**: `minitorch/tensor_data.py`

### Task 2.2: Tensor Broadcasting
**File to Edit**: `minitorch/tensor_data.py`

### Task 2.3: Tensor Operations
**Files to Edit**: `minitorch/tensor_ops.py`, `minitorch/tensor_functions.py`

### Task 2.4: Extend autodifferentiation to work with tensors and broadcasting
**Files to Edit**: `minitorch/tensor_functions.py`

### Task 2.5: Tensor-Based Neural Network Training
**File to Edit**: `project/run_tensor.py`

**Requirements**:
- Train on the first four datasets and record results in README
- Record time per epoch for performance comparison
- Should match functionality of `project/run_scalar.py` but use tensor operations
- To run streamlit, use:
```bash
streamlit run project/app.py -- 2
```
### SIMPLE
EPOCH=500	
LOSS=34.61727009517126
CORRECT=26
TIME PER EPOCH=0.046s

### DIAG
EPOCH=500	
LOSS=17.422916583893382
CORRECT=42
TIME PER EPOCH=0.048s

### SPLIT
EPOCH=500	
LOSS=28.103361779680792
CORRECT=40
TIME PER EPOCH=0.045s

### XOR
EPOCH=500	
LOSS=29.54060592381118
CORRECT=32
TIME PER EPOCH=0.046s

## Testing

See [testing.md](testing.md) for detailed testing instructions.

## Files

This assignment requires the following files from Module 1. You can get these by running:

```bash
python sync_previous_module.py ../Module-1 .
```

The files that will be synced are:

- `minitorch/operators.py`
- `minitorch/module.py`
- `minitorch/autodiff.py`
- `minitorch/scalar.py`
- `project/run_manual.py`
- `project/run_scalar.py`
