#@title Set up the SDE
import torch
import numpy as np
from tqdm import tqdm
from scipy import integrate

def marginal_prob_std(t, sigma=25.0):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

    Returns:
    The standard deviation.
    """
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma=25.0):
    """Compute the diffusion coefficient of our SDE.

    Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

    Returns:
    The vector of diffusion coefficients.
    """
    return sigma**t

#@title Define the Euler-Maruyama sampler (double click to expand or collapse)

## The number of sampling steps.
num_steps =  500#@param {'type':'integer'}
def Euler_Maruyama_sampler(score_model,
                            num_steps=num_steps,
                            cond_text_embeddings=None,
                            null_text_embeddings=None,
                            eps=1e-3,
                            cfg_scale=4.0):
    """Generate samples from score-based models with the Euler-Maruyama solver.

    Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps.
        Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.

    Returns:
    Samples.
    """

    assert cond_text_embeddings is not None, "cond_text_embedding must be provided"
    device = cond_text_embeddings.device
    t = torch.ones(1, device=device)
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    init_x = torch.randn(1, 1, 32, 32, device=device) * marginal_prob_std(t)[:, None, None, None]

    x = init_x
    if null_text_embeddings is None:
        null_text_embeddings = torch.zeros_like(cond_text_embeddings)

    with torch.no_grad():
        for time_step in time_steps:
            batch_time_step = torch.ones(1, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            cond_score = score_model(x, batch_time_step, cond_text_embeddings)
            uncond_score = score_model(x, batch_time_step, null_text_embeddings)
            score = uncond_score + cfg_scale * (cond_score - uncond_score)
            mean_x = x + (g**2)[:, None, None, None] * score * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
        # Do not include any noise in the last sampling step.
        return mean_x
    


error_tolerance = 1e-5

@torch.no_grad()
def ode_sampler_cfg_label(
    score_model,
    batch_size=16,
    input_size=32,
    in_channel=1,
    null_id=10,
    guidance_scale=4.0,
    atol=error_tolerance,
    rtol=error_tolerance,
    eps=1e-3,
):
    score_model.eval()
    device = next(score_model.parameters()).device
    t1 = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, in_channel, input_size, input_size, device=device) * marginal_prob_std(t1)[:, None, None, None]

    cond = torch.randint(0, null_id - 1, (batch_size,), device=device)
        
    cond_null = torch.full_like(cond, null_id)  # 无条件

    shape = init_x.shape  # (B, 1, 32, 32)

    def score_eval_wrapper(sample_flat, time_steps_np):
        sample = torch.tensor(sample_flat, device=device, dtype=torch.float32).reshape(shape)
        t = torch.tensor(time_steps_np, device=device, dtype=torch.float32).reshape((shape[0],))

        # s_cond, s_uncond
        s_cond   = score_model(sample, t, cond)
        s_uncond = score_model(sample, t, cond_null)
        s_cfg = (1.0 + guidance_scale) * s_cond - guidance_scale * s_uncond

        return s_cfg.detach().cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t_scalar, x_flat):
        time_steps_np = np.full((shape[0],), t_scalar, dtype=np.float32)
        gt = diffusion_coeff(torch.tensor(t_scalar, dtype=torch.float32, device=device)).item()
        return -0.5 * (gt ** 2) * score_eval_wrapper(x_flat, time_steps_np)

    res = integrate.solve_ivp(
        ode_func,
        t_span=(1.0, eps),
        y0=init_x.reshape(-1).cpu().numpy(),
        rtol=rtol,
        atol=atol,
        method='RK45',
    )
    x = torch.tensor(res.y[:, -1], device=device, dtype=torch.float32).reshape(shape)
    return x


@torch.no_grad()
def ode_sampler_cfg_text(
    score_model,
    cond,
    uncond,
    cond_attn_mask=None,
    uncond_attn_mask=None,
    batch_size=16,
    in_channel=4,
    guidance_scale=7.0,
    atol=error_tolerance,
    rtol=error_tolerance,
    eps=1e-3,
):
    score_model.eval()
    device = next(score_model.parameters()).device
    t1 = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, in_channel, 64, 64, device=device) * marginal_prob_std(t1)[:, None, None, None]

    cond = cond.repeat(batch_size, 1, 1)
    uncond = uncond.repeat(batch_size, 1, 1)
    shape = init_x.shape  # (B, 1, 32, 32)

    def score_eval_wrapper(sample_flat, time_steps_np):
        sample = torch.tensor(sample_flat, device=device, dtype=torch.float32).reshape(shape)
        t = torch.tensor(time_steps_np, device=device, dtype=torch.float32).reshape((shape[0],))

        # s_cond, s_uncond
        s_cond   = score_model(sample, t, cond, cond_attn_mask)
        s_uncond = score_model(sample, t, uncond, uncond_attn_mask)
        s_cfg = (1.0 + guidance_scale) * s_cond - guidance_scale * s_uncond

        return s_cfg.detach().cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t_scalar, x_flat):
        time_steps_np = np.full((shape[0],), t_scalar, dtype=np.float32)
        gt = diffusion_coeff(torch.tensor(t_scalar, dtype=torch.float32, device=device)).item()
        return -0.5 * (gt ** 2) * score_eval_wrapper(x_flat, time_steps_np)

    res = integrate.solve_ivp(
        ode_func,
        t_span=(1.0, eps),
        y0=init_x.reshape(-1).cpu().numpy(),
        rtol=rtol,
        atol=atol,
        method='RK45',
    )
    x = torch.tensor(res.y[:, -1], device=device, dtype=torch.float32).reshape(shape)
    return x
