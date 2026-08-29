"""
Constrained Least Squares Example using PyTorch KKT Solver

This example demonstrates solving a constrained least-squares problem using
the pytorchLagrangeOpt function from batchoptim.py.

Problem: Fit a line y = m*x + c to noisy data points
Subject to: slope and intercept constraints

The batchoptim.py framework handles:
- Batch optimization (multiple samples simultaneously)
- KKT Newton step for constrained optimization
- Augmented Lagrangian method as an alternative
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# Import the optimization functions from batchoptim.py
# Assuming batchoptim.py is in the same directory
from batchoptim import pytorchLagrangeOpt, testBatchJac


def create_line_fitting_problem(n_points=20, noise_std=0.2, batch_size=1):
    """
    Create a line fitting problem with constraints.
    
    Returns:
        objF: Objective function (least squares)
        cstF: Constraint function (slope and intercept bounds)
        x0: Initial guess
        true_params: True parameters for comparison
    """
    
    # Generate synthetic data
    np.random.seed(42)
    x_data = np.linspace(-2, 2, n_points)
    true_m, true_c = 0.5, 1.0
    y_data = true_m * x_data + true_c + noise_std * np.random.randn(n_points)
    
    # Convert to PyTorch tensors
    x_data_t = torch.tensor(x_data, dtype=torch.float32).unsqueeze(1)  # (n_points, 1)
    y_data_t = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)  # (n_points, 1)
    
    # Objective function: minimize ||A*x - b||^2
    # x = [m, c]  (slope, intercept)
    # A = [[x_i, 1]] for each data point
    # b = y_i
    def objF(x):
        """
        Objective function: least squares residual.
        
        Args:
            x: (batch_size, 2) tensor of [m, c] parameters
        
        Returns:
            F: (batch_size, n_points) tensor of residuals (A*x - b)
            J: (batch_size, n_points, 2) Jacobian of residuals
        """
        # Ensure x is the right shape
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        # Reshape for batch processing
        # x: (batch_size, 2) -> each row is [m, c]
        m = x[:, 0].unsqueeze(1)  # (batch_size, 1)
        c = x[:, 1].unsqueeze(1)  # (batch_size, 1)
        
        # A matrix: (batch_size, n_points, 2)
        # Each row: [x_i, 1]
        A = torch.stack([x_data_t.squeeze(1)] * batch_size, dim=0)  # (batch_size, n_points)
        A = torch.stack([A, torch.ones_like(A)], dim=2)  # (batch_size, n_points, 2)
        
        # Predicted values: y_pred = m*x_i + c
        # (batch_size, n_points)
        y_pred = m * x_data_t.squeeze(1).unsqueeze(0) + c
        
        # Residuals: (batch_size, n_points)
        F = y_pred - y_data_t.squeeze(1).unsqueeze(0)
        
        # Jacobian: (batch_size, n_points, 2)
        # dF/dm = x_i, dF/dc = 1
        J = torch.stack([
            x_data_t.squeeze(1).unsqueeze(0).expand(batch_size, -1),  # dF/dm = x_i
            torch.ones(batch_size, n_points)                          # dF/dc = 1
        ], dim=2)
        
        return F, J
    
    # Constraint function: C(x) >= 0
    # Constraints:
    #   1. m >= -1  =>  -1 - m <= 0  =>  C0 = -1 - m
    #   2. m <= 1   =>  m - 1 <= 0   =>  C1 = m - 1
    #   3. c >= 0.5 =>  0.5 - c <= 0 =>  C2 = 0.5 - c
    def cstF(x):
        """
        Constraint function: equality constraints C(x) = 0.
        
        For inequality constraints, we convert to equality by introducing
        slack variables or using active set methods.
        
        Here we use: C(x) = max(0, constraint_violation) as an equality.
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        m = x[:, 0]  # slope
        c = x[:, 1]  # intercept

        C = torch.stack([
            m + 1.0,     # m >= -1  => m + 1 >= 0
            1.0 - m,     # m <= 1   => 1 - m >= 0
            c - 0.5      # c >= 0.5 => c - 0.5 >= 0
        ], dim=1)
        
        # Jacobian of constraints: (batch_size, 3, 2)
        # dC/dm and dC/dc
        A = torch.zeros(batch_size, 3, 2)
        A[:, 0, 0] = 1.0   # d(m+1)/dm = 1
        A[:, 1, 0] = -1.0  # d(1-m)/dm = -1
        A[:, 2, 1] = 1.0   # d(c-0.5)/dc = 1
        
        return C, A
    
    # Initial guess: [m, c]
    x0 = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    
    true_params = (true_m, true_c)
    
    return objF, cstF, x0, true_params, x_data, y_data


def example_line_fitting_kkt():
    """
    Solve line fitting problem using KKT Newton method.
    """
    print("=" * 70)
    print("Constrained Least Squares Example (KKT Method)")
    print("=" * 70)
    
    # Create the problem
    objF, cstF, x0, true_params, x_data, y_data = create_line_fitting_problem()
    
    print(f"True parameters: m = {true_params[0]:.4f}, c = {true_params[1]:.4f}")
    print(f"Constraints: m >= -1, m <= 1, c >= 0.5")
    print("-" * 70)
    
    # Run optimization using KKT method
    # Parameters:
    #   objF: objective function returning (F, J)
    #   cstF: constraint function returning (C, A)
    #   x0: initial guess
    #   nIt: number of iterations
    #   lambd: regularization parameter
    #   rho: penalty parameter for augmented Lagrangian
    #   kktP: True for KKT method, False for augmented Lagrangian
    print("Running KKT optimization...")
    x_opt = pytorchLagrangeOpt(
        objF=objF,
        cstF=cstF,
        x0=x0,
        nIt=20,
        lambd=1e-6,
        rho=10.0,
        kktP=True,  # Use KKT method
        verbP=True  # Print progress
    )
    
    # Extract optimized parameters
    m_opt = x_opt[0, 0].item()
    c_opt = x_opt[0, 1].item()
    
    print("-" * 70)
    print("Results (KKT Method):")
    print(f"  Optimal slope (m): {m_opt:.6f}")
    print(f"  Optimal intercept (c): {c_opt:.6f}")
    print(f"  True slope (m): {true_params[0]:.6f}")
    print(f"  True intercept (c): {true_params[1]:.6f}")
    
    # Verify constraints
    print("\nConstraint Verification:")
    print(f"  m >= -1: {m_opt:.4f} >= -1 {'✓' if m_opt >= -1 else '✗'}")
    print(f"  m <= 1:  {m_opt:.4f} <= 1  {'✓' if m_opt <= 1 else '✗'}")
    print(f"  c >= 0.5: {c_opt:.4f} >= 0.5 {'✓' if c_opt >= 0.5 else '✗'}")
    
    # Plot results
    plot_results(x_data, y_data, m_opt, c_opt, true_params, "KKT Method")
    
    return x_opt


def example_line_fitting_alm():
    # Solve line fitting problem using Augmented Lagrangian method.
    print("=" * 70)
    print("Constrained Least Squares Example (Augmented Lagrangian Method)")
    print("=" * 70)
    
    # Create the problem
    objF, cstF, x0, true_params, x_data, y_data = create_line_fitting_problem()
    
    print(f"True parameters: m = {true_params[0]:.4f}, c = {true_params[1]:.4f}")
    print(f"Constraints: m >= -1, m <= 1, c >= 0.5")
    print("-" * 70)
    
    # Run optimization using Augmented Lagrangian method
    x_opt = pytorchLagrangeOpt(
        objF=objF,
        cstF=cstF,
        x0=x0,
        nIt=20,
        lambd=0.0,
        rho=10.0,
        kktP=False,  # Use Augmented Lagrangian method
        verbP=True
    )
    
    # Extract optimized parameters
    m_opt = x_opt[0, 0].item()
    c_opt = x_opt[0, 1].item()
    
    print("-" * 70)
    print("Results (Augmented Lagrangian):")
    print(f"  Optimal slope (m): {m_opt:.6f}")
    print(f"  Optimal intercept (c): {c_opt:.6f}")
    print(f"  True slope (m): {true_params[0]:.6f}")
    print(f"  True intercept (c): {true_params[1]:.6f}")
    
    # Verify constraints
    print("\nConstraint Verification:")
    print(f"  m >= -1: {m_opt:.4f} >= -1 {'✓' if m_opt >= -1 else '✗'}")
    print(f"  m <= 1:  {m_opt:.4f} <= 1  {'✓' if m_opt <= 1 else '✗'}")
    print(f"  c >= 0.5: {c_opt:.4f} >= 0.5 {'✓' if c_opt >= 0.5 else '✗'}")
    
    # Plot results
    plot_results(x_data, y_data, m_opt, c_opt, true_params, "Augmented Lagrangian")
    
    return x_opt


def plot_results(x_data, y_data, m_opt, c_opt, true_params, method_name):
    """
    Plot the fitted line against the true line and data points.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # True line
    x_range = np.linspace(-2.5, 2.5, 100)
    y_true = true_params[0] * x_range + true_params[1]
    ax.plot(x_range, y_true, 'g--', label='True line', linewidth=2)
    
    # Fitted line
    y_fit = m_opt * x_range + c_opt
    ax.plot(x_range, y_fit, 'r-', label=f'Fitted line ({method_name})', linewidth=2)
    
    # Data points
    ax.scatter(x_data, y_data, color='blue', alpha=0.6, label='Data points')
    
    # Formatting
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Constrained Least Squares: Line Fitting ({method_name})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add constraint information
    ax.text(0.05, 0.95, f'Constraints: m ∈ [-1, 1], c ≥ 0.5',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.text(0.05, 0.88, f'Optimal m = {m_opt:.4f}, c = {c_opt:.4f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    return fig, ax


def test_gradients():
    """
    Test the gradient computations using the testBatchJac function.
    """
    print("=" * 70)
    print("Testing Gradient Computations")
    print("=" * 70)
    
    objF, cstF, x0, _, _, _ = create_line_fitting_problem()
    
    print("\nTesting objective function Jacobian:")
    testBatchJac(objF, x0, eps=1e-6)
    
    print("\nTesting constraint function Jacobian:")
    testBatchJac(cstF, x0, eps=1e-6)


def batch_example():
    """
    Example with multiple batch elements simultaneously.
    """
    print("=" * 70)
    print("Batch Processing Example")
    print("=" * 70)
    
    # Create multiple initial guesses for batch processing
    objF, cstF, x0_single, true_params, x_data, y_data = create_line_fitting_problem()
    
    # Batch of initial guesses
    x0_batch = torch.tensor([
        [0.0, 0.0],   # Guess 1
        [1.0, 0.5],   # Guess 2
        [-0.5, 2.0],  # Guess 3
    ], dtype=torch.float32)
    
    print(f"Batch size: {x0_batch.shape[0]}")
    print(f"Initial guesses:\n{x0_batch.numpy()}")
    print("-" * 70)
    
    # Run optimization on all batch elements simultaneously
    x_opt_batch = pytorchLagrangeOpt(
        objF=objF,
        cstF=cstF,
        x0=x0_batch,
        nIt=15,
        lambd=1e-6,
        rho=10.0,
        kktP=True,
        verbP=True
    )
    
    print("-" * 70)
    print("Batch Results:")
    for i in range(x_opt_batch.shape[0]):
        m_opt = x_opt_batch[i, 0].item()
        c_opt = x_opt_batch[i, 1].item()
        print(f"  Batch {i+1}: m = {m_opt:.6f}, c = {c_opt:.6f}")
        print(f"    Constraints: m >= -1: {m_opt >= -1}, m <= 1: {m_opt <= 1}, c >= 0.5: {c_opt >= 0.5}")
    
    return x_opt_batch


if __name__ == "__main__":
    print("Constrained Least Squares Examples")
    
    # Example 1: KKT Method
    print("\n")
    example_line_fitting_kkt()
    
    # Example 2: Augmented Lagrangian Method
    print("\n")
    example_line_fitting_alm()
    
    # Example 3: Batch Processing
    print("\n")
    batch_example()
    
    # Example 4: Test Gradients (uncomment to run)
    # print("\n")
    # test_gradients()
    print("All examples completed!")
