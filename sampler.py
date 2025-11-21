from flow_utils import FlowerSampler, GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta, FlowerCFGTrainer
from models.flow_dit_model import DiT_models
from diffusers.models import AutoencoderKL
import matplotlib.pyplot as plt
from flow_utils import EulerSimulator, CFGVectorFieldODE
import torch
from torchvision.utils import make_grid
import os
import torch
from models.ddpm_dit_model import DiT_models
from diffusion import create_diffusion
from diffusers import AutoencoderKL
from models.sde_dit_model import DiT_models
from sde_utils.sde_utils import *
def flower_flow_sampler():
    device = "cuda:5"
    latent_size = 256 // 8 
    path = GaussianConditionalProbabilityPath(
        p_data = FlowerSampler(device=device),
        p_simple_shape = [4, latent_size, latent_size],
        alpha = LinearAlpha(),
        beta = LinearBeta()
    ).to(device)

    # Initialize model
    dit = DiT_models["DiT-B/4"](
        in_channels=4,
        learn_sigma=False,
        input_size=latent_size,
        num_classes=102,
    )
    state_dict = torch.load('results/flower_flow/DiT_epoch_999.pt',weights_only=False, map_location='cpu')
    dit.load_state_dict(state_dict)
    dit = dit.to(device)
    vae = AutoencoderKL.from_pretrained("ckpts/autokl").to(device)
    
    num_classes = 4
    samples_per_class = 4
    num_timesteps = 500
    guidance_scales = [0.5, 1.0, 4.0]

    # Graph
    fig, axes = plt.subplots(1, len(guidance_scales), figsize=(10 * len(guidance_scales), 10))
    for idx, w in enumerate(guidance_scales):
        # Setup ode and simulator
        ode = CFGVectorFieldODE(dit, guidance_scale=w)
        simulator = EulerSimulator(ode)

        # Sample initial conditions
        # y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.int64).repeat_interleave(samples_per_class).to(device)
        y = torch.randint(0, 102, (num_classes,), dtype=torch.int64).to(device).repeat_interleave(samples_per_class).to(device)
        
        num_samples = y.shape[0]
        x0, _ = path.p_simple.sample(num_samples)
        # Simulate
        ts = torch.linspace(0,1,num_timesteps).view(1, -1, 1, 1, 1).expand(num_samples, -1, 1, 1, 1).to(device)
        x1 = simulator.simulate(x0, ts, y=y)

        # Plot
        x1 = vae.decode(x1 / 0.18215).sample
        x1 = x1.detach().cpu()
        grid = make_grid(x1, nrow=samples_per_class, normalize=True, value_range=(-1,1))
        axes[idx].imshow(grid.permute(1, 2, 0).cpu())
        axes[idx].axis("off")
        axes[idx].set_title(f"Guidance: $w={w:.1f}$", fontsize=25)
    os.makedirs('results/flower_flow_sample', exist_ok=True)
    plt.savefig(f'results/flower_flow_sample/sample.png')
    plt.close(fig)

def flower_ddpm_sampler():
    device = "cuda:7"
    latent_size = 256 // 8
    model = DiT_models["DiT-B/4"](
        in_channels=4,
        input_size=latent_size,
        num_classes=102
    )
    model.to(device)
    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained("ckpts/autokl").to(device)
    state_dict = torch.load('results/flower_ddpm/checkpoints/0175000.pt',weights_only=False, map_location='cpu')
    model.load_state_dict(state_dict['model'])

    model.train()

        
    with torch.no_grad():
        model.eval()
        guidance_scales = [0.5, 1.0, 4.0]

    # Graph
        fig, axes = plt.subplots(1, len(guidance_scales), figsize=(10 * len(guidance_scales), 10))
        for idx, w in enumerate(guidance_scales):
            z = torch.randn(16, 4, latent_size, latent_size, device=device)
            y = torch.randint(0, 102, (16,), device=device)

            # Setup classifier-free guidance:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([102] * 16, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=w)
            images = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
            images, _ = images.chunk(2, dim=0)  # Remove null class samples
            images = vae.decode(images / 0.18215).sample

            grid = make_grid(images, nrow=4, normalize=True, value_range=(-1,1))
            axes[idx].imshow(grid.permute(1, 2, 0).cpu())
            axes[idx].axis("off")
            axes[idx].set_title(f"Guidance: $w={w:.1f}$", fontsize=25)
        # os.makedirs('results/flower_ddpm_sample', exist_ok=True)
        plt.savefig(f'results/flowers_samples_ddpm.png')
        plt.close(fig)

def flower_sde_sampler():
    device = "cuda:7"
    latent_size = 256 // 8
    model = DiT_models["DiT-B/4"](
        in_channels=4,
        input_size=latent_size,
        num_classes=102
    )
    model.to(device)
    vae = AutoencoderKL.from_pretrained("ckpts/autokl").to(device)
    state_dict = torch.load('results/flower_sde/checkpoints/0238500.pt',weights_only=False, map_location='cpu')
    model.load_state_dict(state_dict['model'])

    model.train()

        
    with torch.no_grad():
        model.eval()
        guidance_scales = [0.5, 1.0, 4.0]

    # Graph
        fig, axes = plt.subplots(1, len(guidance_scales), figsize=(10 * len(guidance_scales), 10))
        for idx, w in enumerate(guidance_scales):
            images = ode_sampler_cfg_label(model, batch_size=16, in_channel=4, null_id=102, guidance_scale=w)
            images = vae.decode(images / 0.18215).sample

            grid = make_grid(images, nrow=4, normalize=True, value_range=(-1,1))
            axes[idx].imshow(grid.permute(1, 2, 0).cpu())
            axes[idx].axis("off")
            axes[idx].set_title(f"Guidance: $w={w:.1f}$", fontsize=25)
        # os.makedirs('results/flower_ddpm_sample', exist_ok=True)
        plt.savefig(f'assets/flowers_samples_sde.png')
        plt.close(fig)


if __name__ == "__main__":
    # flower_flow_sampler()
    # flower_ddpm_sampler()
    flower_sde_sampler()