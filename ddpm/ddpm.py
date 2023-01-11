from typing import Union

import numpy as np
import torch
from torch import nn


class DDPMSampler(nn.Module):
    def __init__(
        self,
        beta_1: float = 0.001,
        beta_T: float = 0.02,
        device: Union[str, torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = device

        max_num_timesteps = 1000

        self.betas = torch.linspace(
            beta_1, beta_T, max_num_timesteps, device=self.device
        )

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # discrete timesteps
        self.timesteps = torch.from_numpy(
            np.arange(0, max_num_timesteps)[::-1].copy()
        ).to(self.device)

    def _get_sigma(self, t: int) -> torch.Tensor:
        alpha_cumprod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)
        alpha_cumprod_t = self.alphas_cumprod[t]

        # variance for a deterministic x_0 (see section 3.2.)
        variance_t = (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * self.betas[t]
        sigma_t = torch.sqrt(variance_t)

        return sigma_t

    def sample(
        self, eps_theta_t: torch.Tensor, t: torch.Tensor, x_t: torch.Tensor
    ) -> torch.Tensor:
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]

        # See eq. (11)
        mu_theta_t = (
            1
            / alpha_t**0.5
            * (x_t - (1 - alpha_t) / (1 - alpha_cumprod_t) ** 0.5 * eps_theta_t)
        )

        z = (
            torch.randn(x_t.shape, device=self.device)
            if t > 0
            else torch.zeros_like(x_t, device=self.device)
        )
        sigma_t = self._get_sigma(t)

        # See algorithm (2) and eq. (1)
        x_t_prev = mu_theta_t + sigma_t * z

        return x_t_prev
