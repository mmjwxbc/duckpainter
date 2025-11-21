import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)


def make_projection(D, d, device=device):

    A = torch.randn(D, d, device=device)
    Q, _ = torch.linalg.qr(A)  

    return Q  

def sample_underlying_2d(n_points):
    theta = np.linspace(0, 4 * np.pi, n_points)
    r = theta / (4 * np.pi) * 2.0  

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    pts = np.stack([x, y], axis=1)

    pts += 0.02 * np.random.randn(*pts.shape)
    pts = torch.from_numpy(pts).float()
    return pts   


class MLP5(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * 5 + [out_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        """
        x: [B, D]
        t: [B] or [B,1]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)


def train_toy(
    D=16,
    d=2,
    target_type="x",
    n_samples=20000,
    batch_size=256,
    epochs=50,
    lr=1e-3,
):
    
    P = make_projection(D, d)  # [D, 2]
    x_hat = sample_underlying_2d(n_samples).to(device)  # [N, 2]
    x = x_hat @ P.t()  
    
    sigma = x.std()
    # sigma = 1.0 

    dataset = TensorDataset(x)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = MLP5(in_dim=D + 1, hidden_dim=256, out_dim=D).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for step, (x_batch,) in enumerate(loader):
            x_batch = x_batch.to(device)  # [B, D]

            B = x_batch.size(0)
            x_batch = x_batch / sigma
            #  t ~ Uniform(0,1)
            t = torch.rand(B, device=device)[:,None]
            
            noise = torch.randn_like(x_batch)
            x_t = t * x_batch + (1 - t) * noise 
            
            pred = model(x_t, t)

            if target_type == "x":
                dnorm = torch.clamp(1 - t, min=0.05)
                v = (x_batch - x_t) / dnorm
                vp = (pred - x_t) / dnorm
                loss = ((v - vp) ** 2).mean()

            elif target_type == "v":
                loss = ((noise - x_batch - pred) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"[D={D}] Epoch {epoch + 1}/{epochs} | {target_type}-prediction loss: {loss.item():.4f}")

    return model, P, x_hat, x, sigma



def show_point(x, P, x_hat_true, target_type, D, cur_step=None):
    
    pred_2d = x @ P  # [N,2]

    x_hat_np = x_hat_true.cpu().numpy()
    pred_2d_np = pred_2d.cpu().numpy()

    plt.figure(figsize=(5, 5))
    plt.scatter(x_hat_np[:, 0], x_hat_np[:, 1], s=5, alpha=0.3, label="True 2D data")
    plt.scatter(pred_2d_np[:, 0], pred_2d_np[:, 1], s=5, alpha=0.7, label=f"Generated ({target_type}-pred)")
    plt.legend()
    plt.title(f"D={D}, target={target_type}")
    plt.axis("equal")
    plt.tight_layout()
    save_filename = f"toy_D{D}_target_{target_type}_step_{cur_step}.png" if cur_step is not None else f"toy_D{D}_target_{target_type}.png"
    plt.savefig(save_filename, dpi=200)


def visualize_2d(model, P, n_points=2000, target_type="x", steps=250, x_true=None, sigma=1.0):
    
    model.eval()
    P = P.to(device)
    D, d = P.shape  # P: [D,2]

    x = torch.randn(n_points, D, device=device)
    dt = 1.0 / steps

    for i in range(1, steps + 1, 1):
        with torch.no_grad():
            t = torch.full((n_points,), i * dt, device=device) 
            x_t = x                                            

            pred = model(x_t, t)
            
            if target_type == "x":
                vp = (pred - x_t) / (1 - t)[:,None].clamp_min(1e-2)    
            elif target_type == "v":
                vp = pred
            
            x = x_t + dt * vp  
            # if i % 50 == 0 or i == 1:
            #     show_point((x_t - i * dt * vp) * sigma, P, x_hat_true, target_type, D, cur_step=i)

    show_point(x * sigma, P, x_true, target_type, D)

if __name__ == "__main__":
    Ds = [16, 512]
    target_types = ["x", "v"]

    for D in Ds:
        for tt in target_types:
            print(f"\n=== Training D={D}, target={tt} ===")
            model, P, x_hat, x, sigma = train_toy(
                D=D,
                d=2,
                target_type=tt,
                n_samples=20000,
                batch_size=1024,
                epochs=1000, 
                lr=1e-3,
            )
            visualize_2d(model, P, n_points=20000, target_type=tt, x_true=x_hat, sigma=sigma)