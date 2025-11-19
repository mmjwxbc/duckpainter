from flow_utils import FlowerSampler, GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta, FlowerCFGTrainer
from models.flow_dit_model import DiT_models
from diffusers.models import AutoencoderKL
import matplotlib.pyplot as plt
from flow_utils import EulerSimulator, CFGVectorFieldODE
import torch
from torchvision.utils import make_grid
import os
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
    
if __name__ == "__main__":
    flower_flow_sampler()