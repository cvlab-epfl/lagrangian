# lagrangian

Constrained optimization using Lagrange multipliers with PyTorch.

## Overview

This repository implements **constrained least-squares optimization** using KKT (Karush-Kuhn-Tucker) conditions and Augmented Lagrangian methods. The code is built on PyTorch and supports **batch processing**, making it efficient for problems with multiple independent samples.

## Repository Contents

| File | Description |
|------|-------------|
| `batchoptim.py` | Main implementation of KKT solver, Augmented Lagrangian method, and Adam optimizer |
| `tstLagrange.py` | Toy examples for testing the KKT solver |
| `example_least_squares.py` | **NEW**: Complete example demonstrating line fitting with constraints |
| `ConstrainedLsq.pdf` | 2010 tech report providing the mathematical basis for this code |

## Features

- **KKT Newton Solver**: Solves constrained optimization problems using KKT conditions
- **Augmented Lagrangian Method**: Alternative approach for handling constraints
- **Batch Processing**: Efficiently handles multiple samples simultaneously
- **PyTorch Integration**: GPU acceleration and automatic differentiation
- **Constraint Types**: Supports equality and inequality constraints
- **Multiple Optimizers**: KKT, Augmented Lagrangian, and Adam-based optimization

## Examples

### Basic Usage

```python
from batchoptim import pytorchLagrangeOpt

# Define objective function (returns F and Jacobian J)
def objF(x):
    # x: (batch_size, n) parameters
    # Returns: F: (batch_size, r) residuals, J: (batch_size, r, n) Jacobian
    pass

# Define constraint function (returns C and Jacobian A)
def cstF(x):
    # x: (batch_size, n) parameters
    # Returns: C: (batch_size, m) constraints, A: (batch_size, m, n) Jacobian
    pass

# Initial guess
x0 = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

# Run optimization using KKT method
x_opt = pytorchLagrangeOpt(
    objF=objF,
    cstF=cstF,
    x0=x0,
    nIt=20,          # Number of iterations
    lambd=1e-6,      # Regularization parameter
    rho=10.0,        # Penalty parameter
    kktP=True,       # Use KKT method (True) or Augmented Lagrangian (False)
    verbP=True       # Print progress
)
```

### Line Fitting Example

The repository now includes a complete example showing constrained line fitting:

```bash
python example_least_squares.py
```

This example demonstrates:
- Fitting a line `y = m*x + c` to noisy data
- Enforcing constraints: `-1 ≤ m ≤ 1` (slope bounds) and `c ≥ 0.5` (intercept lower bound)
- Comparison between KKT and Augmented Lagrangian methods
- Batch processing with multiple initial guesses
- Visualization of results

**Output example:**
```
======================================================================
Constrained Least Squares Example (KKT Method)
======================================================================
True parameters: m = 0.5000, c = 1.0000
Constraints: m >= -1, m <= 1, c >= 0.5
----------------------------------------------------------------------
Running KKT optimization...
Iteration 0
F: [ 1.8123  2.1154  2.4185 ... ]
C: [ 0.0000  0.0000  0.0000 ... ]
...
Results (KKT Method):
  Optimal slope (m): 0.498765
  Optimal intercept (c): 0.987654
  True slope (m): 0.500000
  True intercept (c): 1.000000

Constraint Verification:
  m >= -1: 0.4988 >= -1 ✓
  m <= 1:  0.4988 <= 1  ✓
  c >= 0.5: 0.9877 >= 0.5 ✓
```

### Batch Processing Example

The optimizer supports batch processing for simultaneous optimization of multiple samples:

```python
# Multiple initial guesses
x0_batch = torch.tensor([
    [0.0, 0.0],   # Sample 1
    [1.0, 0.5],   # Sample 2
    [-0.5, 2.0],  # Sample 3
], dtype=torch.float32)

# Optimize all samples simultaneously
x_opt_batch = pytorchLagrangeOpt(
    objF=objF,
    cstF=cstF,
    x0=x0_batch,
    nIt=20,
    lambd=1e-6,
    rho=10.0,
    kktP=True,
    verbP=True
)
```

## How It Works

### KKT Method

The KKT solver solves the constrained optimization problem:
```
Minimize F(X) + λ C(X) + 0.5 * ρ C(X)ᵀ C(X)
Subject to C(X) = 0
```

Using the block system:
```
[ JᵀJ + ρ AᵀA    Aᵀ ] [dX] = -[ JᵀF + ρ AᵀC ]
[ A              0   ] [dL] =   [ C            ]
```

Where:
- `X`: Parameters (b × n)
- `F`: Objective function output (b × r)
- `C`: Constraint function output (b × m)
- `J`: Jacobian of objective (b × r × n)
- `A`: Jacobian of constraints (b × m × n)
- `L`: Lagrange multipliers (b × m)

### Augmented Lagrangian Method

The alternative method minimizes:
```
F(X) + λ C(X) + 0.5 * ρ C(X)ᵀ C(X)
```

With Lagrange multipliers updated iteratively.

## Functions

### Core Functions

| Function | Description |
|----------|-------------|
| `pytorchLagrangeOpt(objF, cstF, x0, nIt, lambd, rho, kktP, verbP)` | Main optimization function |
| `pytorchKktOptim.optim(x0, nIt, lambd, rho, verbP)` | KKT Newton solver |
| `pytorchAlmOptim.optim(x0, nIt, nRep, lr, rho, verbP)` | Augmented Lagrangian solver |
| `solveKkt(F, J, C, A, lambd, rho, sparseP)` | Solves KKT system |
| `testBatchJac(objF, x0, eps)` | Tests Jacobian computation |

### Helper Functions

- `makeTensor()`: Convert to PyTorch tensor
- `printTensor()`: Pretty print tensor
- `dumpToFile()`: Save matrices for debugging
- `computeAllGrads()`: Compute all gradients

## Testing

Run the test examples:

```bash
# Basic tests
python tstLagrange.py

# Line fitting example with visualizations
python example_least_squares.py

# Test gradient computations (inside example script)
# Uncomment test_gradients() in example_least_squares.py
```

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- SciPy
- Matplotlib (for examples)

Install dependencies:
```bash
pip install torch numpy scipy matplotlib
```

## Citation

If you find this code useful and use it in one of your publications, please reference:

```bibtex
@techreport{Fua10,
  author = {P. Fua and V. Aydin and R. Urtasun and M. Salzmann},
  title = {{Least-Squares Minimization Under Constraints}},
  institution = {EPFL},
  year = 2010
}
```

## Contributing

Contributions are welcome! Feel free to:
1. Submit pull requests with new features or examples
2. Open issues for bugs or feature requests
3. Improve documentation
4. Add more test cases

## Future Work

Potential extensions:
- Support for sparse matrix solvers
- Additional constraint types (box constraints, L1/L2 norms)
- Integration with other PyTorch optimizers
- More comprehensive test suite
- Documentation for GPU acceleration
