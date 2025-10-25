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
SIMPLE
Epoch  10  loss  32.041582644141684 correct 26
Epoch  20  loss  26.807157393085536 correct 40
Epoch  30  loss  19.01792201433944 correct 49
Epoch  40  loss  14.008797357603056 correct 46
Epoch  50  loss  12.716890479325864 correct 45
Epoch  60  loss  8.503467360274403 correct 48
Epoch  70  loss  6.288623394595263 correct 49
Epoch  80  loss  5.261513974275125 correct 50
Epoch  90  loss  4.540012121863494 correct 50
Epoch  100  loss  4.003742805390184 correct 50
Epoch  110  loss  3.582462893333349 correct 50
Epoch  120  loss  3.2422957244383213 correct 50
Epoch  130  loss  2.9622152958839996 correct 50
Epoch  140  loss  2.7390129483503944 correct 50
Epoch  150  loss  2.5420486202398807 correct 50
Epoch  160  loss  2.3716929766349977 correct 50
Epoch  170  loss  2.2226250579731057 correct 50
Epoch  180  loss  2.0897407623524598 correct 50
Epoch  190  loss  1.9724359124405968 correct 50
Epoch  200  loss  1.8581166180541806 correct 50
Epoch  210  loss  1.782637542410291 correct 50
Epoch  220  loss  1.6996094040740353 correct 50
Epoch  230  loss  1.6087657688422166 correct 50
Epoch  240  loss  1.5351783657049134 correct 50
Epoch  250  loss  1.470864316150218 correct 50
Epoch  260  loss  1.4111943221502639 correct 50
Epoch  270  loss  1.3586767699945397 correct 50
Epoch  280  loss  1.303985748100964 correct 50
Epoch  290  loss  1.2552334402278853 correct 50
Epoch  300  loss  1.2205386077372944 correct 50
Epoch  310  loss  1.1817978678367569 correct 50
Epoch  320  loss  1.1387017663934411 correct 50
Epoch  330  loss  1.0920540981873714 correct 50
Epoch  340  loss  1.0633793059796535 correct 50
Epoch  350  loss  1.0223786755559376 correct 50
Epoch  360  loss  0.9897711662421614 correct 50
Epoch  370  loss  0.9625854800024016 correct 50
Epoch  380  loss  0.9338856263591412 correct 50
Epoch  390  loss  0.9063509687093879 correct 50
Epoch  400  loss  0.8771301219592105 correct 50
Epoch  410  loss  0.8548780568676356 correct 50
Epoch  420  loss  0.8316763249885238 correct 50
Epoch  430  loss  0.8084908012115565 correct 50
Epoch  440  loss  0.7867069361362329 correct 50
Epoch  450  loss  0.7636129470675184 correct 50
Epoch  460  loss  0.7456328194686783 correct 50
Epoch  470  loss  0.7264358751504968 correct 50
Epoch  480  loss  0.7079399621331167 correct 50
Epoch  490  loss  0.6881056571299115 correct 50
Epoch  500  loss  0.6728742671384766 correct 50



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
