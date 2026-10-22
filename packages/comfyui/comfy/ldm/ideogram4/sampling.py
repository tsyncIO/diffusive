import math

import torch

from ... import model_sampling

# ideogram-oss/ideogram4 scheduler.LogitNormalSchedule
LOGSNR_MIN = -15.0
LOGSNR_MAX = 18.0
T_MIN = 1.0 / (1.0 + math.exp(0.5 * LOGSNR_MAX))
T_MAX = 1.0 / (1.0 + math.exp(0.5 * LOGSNR_MIN))
# where the reference loop starts (from the noise itself) and stops (decoding the state as it is)
SIGMA_START = float(1 - torch.tensor(T_MIN, dtype=torch.float64).to(torch.float32))
SIGMA_END = float(1 - torch.tensor(T_MAX, dtype=torch.float64).to(torch.float32))


def ideogram4_sigmas(num_steps, width, height, mu, std):
    """The reference loop's sigmas, descending (num_steps + 1): pipeline_ideogram4 steps from reference time
    schedule(u[i + 1]) to schedule(u[i]), u = linspace(0, 1, num_steps + 1) in float32, and sigma = 1 - t. mu plus the
    resolution term is the logit-normal mean, std its spread."""
    mean = mu + 0.5 * math.log((width * height) / (512 * 512))
    u = torch.linspace(0.0, 1.0, num_steps + 1, dtype=torch.float32).to(torch.float64)
    t = (1 - torch.special.expit(mean + std * torch.special.ndtri(u))).clamp(T_MIN, T_MAX).to(torch.float32)
    return (1 - t).flip(0)


class ModelSamplingIdeogram4(model_sampling.ModelSamplingDiscreteFlow, model_sampling.CONST):
    """pipeline_ideogram4 starts from the noise itself at sigma 1 - t_min and decodes the state where it stops, at
    sigma 1 - t_max, without the 1 / (1 - sigma) rescale."""

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        sigma = model_sampling.reshape_sigma(sigma, noise.ndim)
        noise_sigma = torch.where(sigma >= SIGMA_START, torch.ones_like(sigma), sigma)
        return noise_sigma * noise + (1.0 - sigma) * latent_image

    def inverse_noise_scaling(self, sigma, latent):
        if float(sigma) <= SIGMA_END:
            return latent
        return super().inverse_noise_scaling(sigma, latent)
