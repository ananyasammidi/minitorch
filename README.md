
# MiniTorch

## Overview

Minitorch is a lightweight re-implementation of the Torch API completely in Python. It implements essential components of PyTorch like auto-differentiation, tensors, and neural networks from scratch

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
