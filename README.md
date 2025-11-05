
# MiniTorch

## Overview

Module 2 introduces **Tensors** - multidimensional arrays that extend the scalar autodifferentiation system from Module 1. While the scalar system is correct, it's inefficient due to Python overhead. Tensors solve this by grouping operations together and enabling faster implementations.

## Installation

See [installation.md](installation.md) for detailed setup instructions.

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev,extra]"

# Verify installation
python -c "import minitorch; print('Success!')"

# Train tensor-based model
python project/run_tensor.py
```

**Streamlit**:
- To run streamlit, use:
```bash
streamlit run project/app.py -- 2
```
