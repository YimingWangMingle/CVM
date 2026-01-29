import torch
import torch.nn as nn
import torch.nn.functional as F

class CVMEstimator:
    """
    Covariance Volume Maximization (CVM) Estimator.
    Tracks the covariance of latent displacements and computes the exploration bonus.
    
    Formula:
    Sigma_t = (1 - beta) * Sigma_{t-1} + beta * (delta_phi * delta_phi^T) + lambda * I
    Bonus = log(1 + delta_phi^T * Sigma_{t-1}^{-1} * delta_phi)
    """
    def __init__(self, latent_dim: int, beta: float = 0.01, ridge: float = 0.1, device: str = "cpu"):
        self.latent_dim = latent_dim
        self.beta = beta
        self.ridge = ridge
        self.device = device
        
        # Initialize Sigma as Identity * ridge (or just Identity)
        # We store the inverse for efficiency if needed, or just Sigma.
        # Since we need Sigma^{-1} * delta_phi, and rank-1 updates are frequent,
        # we might want to just store Sigma and solve, or use Sherman-Morrison if n is large.
        # For n=256, direct solve is maybe okay but rank-1 update is O(n^2).
        # Let's stick to storing Sigma and using torch.linalg.solve or inverse.
        # Given "EMA update", it's not exactly a pure rank-1 update of the inverse 
        # because of the decay (1-beta).
        # So we update Sigma explicitly.
        
        self.sigma = torch.eye(latent_dim, device=device) * ridge
        
    def to(self, device):
        self.device = device
        self.sigma = self.sigma.to(device)
        return self

    def update(self, delta_phi: torch.Tensor):
        """
        Update covariance estimate with new displacement(s).
        delta_phi: (B, latent_dim) or (latent_dim,)
        """
        if delta_phi.dim() == 1:
            delta_phi = delta_phi.unsqueeze(0)
            
        # We process batch updates by averaging or sequential? 
        # The paper implies online update "At time t". 
        # If batch, we can approximate.
        
        batch_size = delta_phi.shape[0]
        
        # Calculate outer products: (B, n, n)
        outer = torch.bmm(delta_phi.unsqueeze(2), delta_phi.unsqueeze(1))
        # Average over batch if multiple
        outer_mean = outer.mean(dim=0)
        
        # EMA Update
        # Sigma_t = (1 - beta) * Sigma_{t-1} + beta * mean(outer) + lambda * I ?
        # The paper Eq 6: Sigma_t = (1-beta)Sigma_{t-1} + beta * dp * dp^T + lambda * I
        # The lambda * I seems to be added *at each step* or just ensuring positive definiteness?
        # Usually ridge is added to the result, not accumulated. 
        # "add a small ridge term lambda I to ensure Sigma_t is strictly positive definite"
        # If we accumulate lambda I every step, it will blow up or stabilize at lambda/beta?
        # It's likely: Sigma_raw = (1-beta)Sigma_raw + beta * outer
        # Sigma_used = Sigma_raw + lambda * I
        # Let's assume self.sigma tracks the raw covariance.
        
        self.sigma = (1 - self.beta) * self.sigma + self.beta * outer_mean
        
        # Ensure symmetry
        self.sigma = 0.5 * (self.sigma + self.sigma.T)

    def get_bonus(self, delta_phi: torch.Tensor) -> torch.Tensor:
        """
        Compute CVM bonus.
        b_t = log(1 + delta_phi^T * (Sigma + lambda I)^-1 * delta_phi)
        
        delta_phi: (B, latent_dim)
        Returns: (B,)
        """
        if delta_phi.dim() == 1:
            delta_phi = delta_phi.unsqueeze(0)
            
        # Add ridge for stability
        sigma_reg = self.sigma + torch.eye(self.latent_dim, device=self.device) * self.ridge
        
        # Compute x^T * Sigma^-1 * x
        # We can use solve: Sigma * y = x => y = Sigma^-1 * x
        # Then x^T * y
        
        # delta_phi: (B, N)
        # sigma_reg: (N, N)
        # We need (Sigma^-1 @ delta_phi.T).T
        
        # Using torch.linalg.solve(A, B) computes A^-1 B
        # We want Sigma^-1 * delta_phi^T
        # B is (N, Batch)
        
        B_mat = delta_phi.T
        try:
            inv_product = torch.linalg.solve(sigma_reg, B_mat) # (N, B)
        except RuntimeError:
            # Fallback for singular matrix (shouldn't happen with ridge)
            inv_product = torch.linalg.solve(sigma_reg + 1e-4 * torch.eye(self.latent_dim, device=self.device), B_mat)
            
        # Dot product: sum(delta_phi * inv_product.T, dim=1)
        quad_form = (delta_phi * inv_product.T).sum(dim=1)
        
        return torch.log(1 + quad_form)

