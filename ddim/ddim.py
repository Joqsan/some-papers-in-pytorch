from typing import Union

import numpy as np
import torch
from torch import nn


class DDIMSampler(nn.Module):
    def __init__(
        self,
        beta_1: float = 0.001,
        beta_T: float = 0.02,
        eta: float = 0.0,
        num_inference_steps=50,
        device: Union[str, torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.eta = eta
        max_num_timesteps = 1000

        self.betas = torch.linspace(
            beta_1, beta_T, max_num_timesteps, device=self.device
        )

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # discrete timesteps
        self.timesteps = torch.from_numpy(np.arange(0, max_num_timesteps)[::-1].copy())

        if num_inference_steps > max_num_timesteps:
            raise ValueError(
                f"num_inference_steps ({num_inference_steps}) has to be less than or equal to max_num_timesteps ({max_num_timesteps})"
            )

        self.step_ratio = max_num_timesteps // num_inference_steps
        timesteps = (
            (np.arange(0, num_inference_steps) * self.step_ratio).round()[::-1].copy()
        )
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def _get_sigma(self, t: int) -> torch.Tensor:
        t_prev = t - self.step_ratio

        # For t = 0, alpha_prod_t = 1 (see page 5). Extend definition for t < 0
        beta_cumprod_t_prev = (
            1 - self.alphas_cumprod[t_prev] if t_prev > 0 else 1 - torch.tensor(1.0)
        )
        beta_cumprod_t = 1 - self.alphas_cumprod[t]

        # See formula (16)
        variance_t = (
            self.eta**2
            * (beta_cumprod_t_prev / beta_cumprod_t)
            * (1 - beta_cumprod_t / beta_cumprod_t_prev)
        )
        sigma_t = variance_t**0.5

        return sigma_t

    def sample(
        self, eps_theta_t: torch.Tensor, t: torch.Tensor, x_t: torch.Tensor
    ) -> torch.Tensor:

        t_prev = t - self.step_ratio
        alpha_cumprod_t_prev = (
            self.alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0)
        )
        alpha_cumprod_t = self.alphas_cumprod[t]

        sigma_t = self._get_sigma(t)

        # "predicted x_0" in eq. (12)
        x_0_predicted = (
            1
            / alpha_cumprod_t ** (0.5)
            * (x_t - (1 - alpha_cumprod_t) ** (0.5) * eps_theta_t)
        )

        # "direction pointing to x_t" in eq. (12)
        direction_to_x_t = (1 - alpha_cumprod_t_prev - sigma_t**2) ** (
            0.5
        ) * eps_theta_t

        eps_t = torch.randn(x_t.shape, device=self.device)

        # Eq. (12)
        x_t_prev = (
            alpha_cumprod_t_prev ** (0.5) * x_0_predicted
            + direction_to_x_t
            + sigma_t * eps_t
        )

        return x_t_prev
