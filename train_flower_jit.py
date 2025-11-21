from flow_utils import FlowerSampler, GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta, FlowerJiTCFGTrainer
from models.flow_dit_model import DiT_models
if __name__ == "__main__":
    device = "cuda:6"
    image_size = 256
    path = GaussianConditionalProbabilityPath(
        p_data = FlowerSampler(device=device),
        p_simple_shape = [3, image_size, image_size],
        alpha = LinearAlpha(),
        beta = LinearBeta()
    ).to(device)

    # Initialize model
    dit = DiT_models['DiT-S/16'](
        in_channels=3,
        learn_sigma=False,
        input_size=image_size,
        num_classes=102,
    )
    # Initialize trainer
    trainer = FlowerJiTCFGTrainer(path = path, model = dit, eta=0.1, device=device)

    # Train!
    trainer.train(num_epochs = 1000, device=device, lr=1e-4, batch_size=128)