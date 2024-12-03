from typing import Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data

from models.base_model import BaseModel, get_norm_layer, init_net
from models.model_utils import gather

class DenoiseDiffusionModel(BaseModel):

    def __init__(self, conf):
        BaseModel.__init__(self, conf)
        self.eps_model = conf.eps_model
        self.beta = torch.linspace(0.0001, 0.02, conf.n_steps).to(self.device)
        self.alpha = 1.0 - self.beta
        # compute cumulative product
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = conf.n_steps
        self.sigma = self.beta


    # forward-diffusion
    def q_xt_x0(self, x0: torch.tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # compute mean and var of xt according to x0
        # xt= sqrt(at)*x0+ sqrt(1-at)*eps
        maen = gather(self.alpha_bar, t) ** 0.5 * x0
        # (batch_size, 1, 1, 1)
        var = 1 - gather(self.alpha_bar, t)
        return maen, var

    # forward-diffusion
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        # compute xt according mean and var of xt
        if eps is None:
            eps = torch.randn_like(x0)
        maen, var = self.q_xt_x0(x0, t)
        return maen + (var ** 0.5) * eps

    # sampling
    def p_sample(self, xt: torch.tensor, t: torch.Tensor):
        # compute xt-1 according xt
        eps_hat = self.eps_model(xt, t)
        alpha_bar = self.tools.gather(self.alpha_bar, t)
        alpha = self.tools.gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** 0.5
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_hat)
        var = self.tools.gather(self.sigma, t)
        eps = torch.randn(xt.shape, device=xt.device)
        return mean + (var ** 0.5) * eps

    # loss
    # x0, (batch_size, C, H, W);
    def loss(self, x0: torch.tensor, noise: Optional[torch.Tensor] = None):
        # distance between loss
        batch_size = x0.shape[0]
        # (batch_size, )
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        if noise is None: noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps=noise)
        eps_hat = self.eps_model(xt, t)
        return F.mse_loss(noise, eps_hat)

    def forward(self):
        pass

    def set_input(self, input):
        pass

    def optimize_parameters(self):
        pass