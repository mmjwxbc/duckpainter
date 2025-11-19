from flow_utils import CifraSampler, GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta, CFGTrainer
from models.flow_dit_model import DiT_models
# Initialize probability path

if __name__ == "__main__":
    device = "cuda:5"

    path = GaussianConditionalProbabilityPath(
        p_data = CifraSampler(device=device),
        p_simple_shape = [3, 32, 32],
        alpha = LinearAlpha(),
        beta = LinearBeta()
    ).to(device)

    # Initialize model
    dit = DiT_models["DiT-SS/2"](
        learn_sigma=False,
        input_size=32,
        num_classes=10,
    )

    # Initialize trainer
    trainer = CFGTrainer(path = path, model = dit, eta=0.1, device=device)

    # Train!
    trainer.train(num_epochs = 50000, device=device, lr=1e-2, batch_size=64)