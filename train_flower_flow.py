from flow_utils import FlowerSampler, GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta, FlowerCFGTrainer
from models.flow_dit_model import DiT_models
from diffusers.models import AutoencoderKL
if __name__ == "__main__":
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
    vae = AutoencoderKL.from_pretrained("ckpts/autokl").to(device)
    # Initialize trainer
    trainer = FlowerCFGTrainer(path = path, model = dit, eta=0.1, device=device, vae=vae)

    # Train!
    trainer.train(num_epochs = 1000, device=device, lr=1e-4, batch_size=64)