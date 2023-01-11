from torch import nn
import torch
import numpy as np


class DDPMSampler(nn.Module):
    def __init__(
        self,
        beta_1: float = 0.001,
        beta_T: float = 0.02,
        device: str = "cuda",
    ) -> None:
        super().__init__()

        max_num_timesteps = 1000

        self.betas = torch.linspace(beta_1, beta_T, max_num_timesteps)

        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, dim=0)

        # discrete timesteps
        self.timesteps = torch.from_numpy(
            np.arange(0, max_num_timesteps)[::-1].copy()
        ).to(device)

    def _get_sigma(self, t: int) -> torch.Tensor:
        alpha_prod_prev = self.alphas_prod[t - 1] if t > 0 else torch.tensor(1.0)
        alpha_prod_curr = self.alphas_prod[t]

        # variance for a deterministic x_0 (see section 3.2.)
        variance = (1 - alpha_prod_prev) / (1 - alpha_prod_curr) * self.betas[t]
        sigma = torch.sqrt(variance)

        return sigma

    def sample(
        self, eps_theta_t: torch.Tensor, t: torch.Tensor, x_t: torch.Tensor
    ) -> torch.Tensor:
        alpha_t = self.alphas[t]
        alpha_prod_t = self.alphas_prod[t]

        # See formula (11)
        mu_theta_t = (
            1
            / alpha_t**0.5
            * (x_t - (1 - alpha_t) / (1 - alpha_prod_t) ** 0.5 * eps_theta_t)
        )

        z = torch.randn(x_t.shape) if t > 0 else torch.zeros_like(x_t)
        z = z.to(x_t.device)

        sigma_t = self._get_sigma(t)
        sigma_t = sigma_t.to(x_t.device)

        # See algorithm (2) and formula (1)
        x_t_prev = mu_theta_t + sigma_t * z

        return x_t_prev
