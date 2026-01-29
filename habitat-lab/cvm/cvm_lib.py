import torch
import torch.nn as nn
import torch.nn.functional as F

class CVMEstimator:
    """
    Covariance Volume Maximization (CVM) Estimator.
    Optimizes the global coverage objective: log det(Sigma_T).
    """
    def __init__(self, latent_dim: int, beta: float = 0.01, ridge: float = 0.1, device: str = "cpu"):
        self.latent_dim = latent_dim
        self.beta = beta  # EMA step size
        self.ridge = ridge  # Ridge term lambda for PD guarantee
        self.device = device
        
        # Initialize Sigma with ridge term
        self.sigma = torch.eye(latent_dim, device=device) * ridge
        
    def to(self, device):
        self.device = device
        self.sigma = self.sigma.to(device)
        return self

    def update(self, delta_phi: torch.Tensor):
        """
        Update covariance estimate using Exponential Moving Average (EMA).
        Sigma_t = (1 - beta) * Sigma_{t-1} + beta * (dp * dp^T) + lambda * I
        """
        if delta_phi.dim() == 1:
            delta_phi = delta_phi.unsqueeze(0)
            
        batch_size = delta_phi.shape[0]
        
        # Compute the outer product of latent displacements
        outer = torch.bmm(delta_phi.unsqueeze(2), delta_phi.unsqueeze(1))
        outer_mean = outer.mean(dim=0)
        
        # EMA update according to Equation 10
        # We integrate ridge term lambda*I into the update to ensure Sigma is always PD
        self.sigma = (1 - self.beta) * self.sigma + self.beta * outer_mean + (self.beta * self.ridge * torch.eye(self.latent_dim, device=self.device))
        
        # Ensure symmetry to prevent numerical drift
        self.sigma = 0.5 * (self.sigma + self.sigma.T)

    def get_bonus(self, delta_phi: torch.Tensor) -> torch.Tensor:
        """
        Compute CVM exploration bonus: b_t = log(1 + delta_phi^T * Sigma_{t-1}^-1 * delta_phi)
        This coincides with the D-optimal design criterion.
        """
        if delta_phi.dim() == 1:
            delta_phi = delta_phi.unsqueeze(0)
            
        # Sigma_t is already regularized during update, but we use Cholesky for stability
        # sigma_reg: (N, N), delta_phi: (B, N)
        try:
            # Solving via Cholesky decomposition as suggested in implementation details
            L = torch.linalg.cholesky(self.sigma)
            # inv_product = Sigma^-1 * delta_phi^T
            inv_product = torch.cholesky_solve(delta_phi.T, L) # (N, B)
        except RuntimeError:
            # Fallback for extreme cases
            sigma_stable = self.sigma + 1e-4 * torch.eye(self.latent_dim, device=self.device)
            inv_product = torch.linalg.solve(sigma_stable, delta_phi.T)
            
        # Quadratic form: delta_phi^T * Sigma^-1 * delta_phi
        quad_form = (delta_phi * inv_product.T).sum(dim=1)
        
        # Final log-determinant gain
        return torch.log(1.0 + quad_form)
