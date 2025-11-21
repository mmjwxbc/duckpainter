from .base import *
from .path import GaussianConditionalProbabilityPath
from .simulator import EulerSimulator
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.datasets import Flowers102
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
class Trainer(ABC):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.AdamW(self.model.parameters(), lr=lr)

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs) -> torch.Tensor:
        # Report model size
        size_b = model_size_b(self.model)
        print(f'Training model with size: {size_b / MiB:.3f} MiB')
        
        # Start
        self.model.to(device)
        self.opt = self.get_optimizer(lr)
        self.model.train()
        self.scheduler = CosineAnnealingLR(self.opt, T_max=num_epochs)

        

class CifarCFGTrainer(Trainer):
    def __init__(self, path: GaussianConditionalProbabilityPath, model: ConditionalVectorField, eta: float, device: str, **kwargs):
        assert eta > 0 and eta < 1
        super().__init__(model, **kwargs)
        self.eta = eta
        self.path = path
        self.device = device

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # Step 1: Sample z,y from p_data
        z, y = self.path.p_data.sample(batch_size)
        # Step 2: Set each label to 10 (i.e., null) with probability eta
        probs = torch.full((batch_size,), self.eta)

        masks = torch.bernoulli(probs).to(self.device)
        y = torch.where(masks==1, y, 10).to(z.device)
        # Step 3: Sample t and x
        t = torch.rand(batch_size,1,1,1).to(z.device)
        x = self.path.sample_conditional_path(z, t)
        # Step 4: Regress and output loss
        return torch.nn.functional.mse_loss(self.model(x, t, y), self.path.conditional_vector_field(x, z, t)) / batch_size
    
    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs) -> torch.Tensor:
        super().train(num_epochs, device, lr, **kwargs)
        pbar = tqdm(range(num_epochs), total=num_epochs)
        for epoch in pbar:
            self.model.train()
            self.opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            self.opt.step()
            self.scheduler.step()
            pbar.set_description(f'Epoch {epoch}, loss: {loss.item():.8f}')
            if epoch % 2000 == 0:
                self.model.eval()
                os.makedirs('results/cifar_flow/', exist_ok=True)
                torch.save(self.model.state_dict(), f'results/cifar_flow/{self.model.__class__.__name__}_epoch_{epoch}.pt')
                # Play with these!
                samples_per_class = 4
                num_timesteps = 500
                guidance_scales = [1.0, 3.0, 5.0]

                # Graph
                fig, axes = plt.subplots(1, len(guidance_scales), figsize=(10 * len(guidance_scales), 10))
                for idx, w in enumerate(guidance_scales):
                    # Setup ode and simulator
                    ode = CFGVectorFieldODE(self.model, guidance_scale=w)
                    simulator = EulerSimulator(ode)

                    # Sample initial conditions
                    # y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.int64).repeat_interleave(samples_per_class).to(device)
                    y = torch.randint(0, 10, (4,), dtype=torch.int64).to(device).repeat_interleave(samples_per_class).to(device)
                    
                    num_samples = y.shape[0]
                    x0, _ = self.path.p_simple.sample(num_samples) # (num_samples, 1, 32, 32)
                    # Simulate
                    ts = torch.linspace(0,1,num_timesteps).view(1, -1, 1, 1, 1).expand(num_samples, -1, 1, 1, 1).to(device)
                    x1 = simulator.simulate(x0, ts, y=y)

                    # Plot
                    grid = make_grid(x1, nrow=samples_per_class, normalize=True, value_range=(-1,1))
                    axes[idx].imshow(grid.permute(1, 2, 0).cpu())
                    axes[idx].axis("off")
                    axes[idx].set_title(f"Guidance: $w={w:.1f}$", fontsize=25)
                plt.savefig(f'results/cifar_flow/{self.model.__class__.__name__}_epoch_{epoch}.png')

            
class FlowerCFGTrainer(Trainer):
    def __init__(self, path: GaussianConditionalProbabilityPath, model: ConditionalVectorField, eta: float, device: str, vae, **kwargs):
        assert eta > 0 and eta < 1
        super().__init__(model, **kwargs)
        self.eta = eta
        self.path = path
        self.device = device
        self.vae = vae

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # Step 1: Sample z,y from p_data
        z, y = self.path.p_data.sample(batch_size)
        # with torch.no_grad():
        #     z = self.vae.encode(z).latent_dist.sample().mul_(0.18215)
        # Step 2: Set each label to 10 (i.e., null) with probability eta
        probs = torch.full((batch_size,), self.eta)

        masks = torch.bernoulli(probs).to(self.device)
        y = torch.where(masks==1, y, 10).to(z.device)
        # Step 3: Sample t and x
        t = torch.rand(batch_size,1,1,1).to(z.device)
        x = self.path.sample_conditional_path(z, t)
        # Step 4: Regress and output loss
        return torch.nn.functional.mse_loss(self.model(x, t, y), self.path.conditional_vector_field(x, z, t)) / batch_size
    
    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs) -> torch.Tensor:
        super().train(num_epochs, device, lr, **kwargs)
        pbar = tqdm(range(num_epochs), total=num_epochs)
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])

        train = Flowers102("data/", split="train", download=True, transform=transform)
        val   = Flowers102("data/", split="val",   transform=transform)
        test  = Flowers102("data/", split="test",  transform=transform)

        dataset = ConcatDataset([train, val, test])
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4,drop_last=True)
        batch_size = 64
        # with profile(
        #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True
        # ) as prof:
        step = 0
        for epoch in pbar:
            for z, y in dataloader:
                z = z.to(device)
                y = y.to(device)
                with torch.no_grad():
                    z = self.vae.encode(z).latent_dist.sample().mul_(0.18215)
        # with record_function("train1_step"):
                self.opt.zero_grad()
                # loss = self.get_train_loss(**kwargs)
                probs = torch.full((batch_size,), self.eta)

                masks = torch.bernoulli(probs).to(self.device)
                y = torch.where(masks==1, y, 10).to(z.device)
                # Step 3: Sample t and x
                t = torch.rand(batch_size,1,1,1).to(z.device)
                # Step 4: Regress and output loss

                x = self.path.sample_conditional_path(z, t)
                    
                loss = torch.nn.functional.mse_loss(self.model(x, t, y), self.path.conditional_vector_field(x, z, t)) / batch_size
                loss.backward()
                self.opt.step()
                self.scheduler.step()
                pbar.set_description(f'Step {step}, loss: {loss.item():.8f}')
                step += 1
                
            # prof.step()
            if epoch % 1 == 0 and epoch > 0:
                os.makedirs('results/flower_flow/', exist_ok=True)
                torch.save(self.model.state_dict(), f'results/flower_flow/{self.model.__class__.__name__}_epoch_{epoch}.pt')
                # Play with these!
                samples_per_class = 1
                num_timesteps = 500
                guidance_scales = [1.0, 3.0, 5.0]

                # Graph
                fig, axes = plt.subplots(1, len(guidance_scales), figsize=(10 * len(guidance_scales), 10))
                for idx, w in enumerate(guidance_scales):
                    # Setup ode and simulator
                    ode = CFGVectorFieldODE(self.model, guidance_scale=w)
                    simulator = EulerSimulator(ode)

                    # Sample initial conditions
                    # y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.int64).repeat_interleave(samples_per_class).to(device)
                    y = torch.randint(0, 102, (1,), dtype=torch.int64).to(device).repeat_interleave(samples_per_class).to(device)
                    
                    num_samples = y.shape[0]
                    x0, _ = self.path.p_simple.sample(num_samples) # (num_samples, 1, 32, 32)
                    # Simulate
                    ts = torch.linspace(0,1,num_timesteps).view(1, -1, 1, 1, 1).expand(num_samples, -1, 1, 1, 1).to(device)
                    x1 = simulator.simulate(x0, ts, y=y)

                    # Plot
                    x1 = self.vae.decode(x1 / 0.18215).sample
                    x1 = x1.detach().cpu()
                    grid = make_grid(x1, nrow=samples_per_class, normalize=True, value_range=(-1,1))
                    axes[idx].imshow(grid.permute(1, 2, 0).cpu())
                    axes[idx].axis("off")
                    axes[idx].set_title(f"Guidance: $w={w:.1f}$", fontsize=25)
                plt.savefig(f'results/flower_flow/{self.model.__class__.__name__}_epoch_{epoch}.png')
                plt.close(fig)
        # print(prof.key_averages().table(sort_by="cpu_time_total"))

class FlowerJiTCFGTrainer(Trainer):
    def __init__(self, path: GaussianConditionalProbabilityPath, model: ConditionalVectorField, eta: float, device: str, **kwargs):
        assert eta > 0 and eta < 1
        super().__init__(model, **kwargs)
        self.eta = eta
        self.path = path
        self.device = device

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # Step 1: Sample z,y from p_data
        z, y = self.path.p_data.sample(batch_size)
        # with torch.no_grad():
        #     z = self.vae.encode(z).latent_dist.sample().mul_(0.18215)
        # Step 2: Set each label to 10 (i.e., null) with probability eta
        probs = torch.full((batch_size,), self.eta)

        masks = torch.bernoulli(probs).to(self.device)
        y = torch.where(masks==1, y, 10).to(z.device)
        # Step 3: Sample t and x
        t = torch.rand(batch_size,1,1,1).to(z.device)
        x = self.path.sample_conditional_path(z, t)
        # Step 4: Regress and output loss
        return torch.nn.functional.mse_loss(self.model(x, t, y), self.path.conditional_vector_field(x, z, t)) / batch_size
    
    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs) -> torch.Tensor:
        super().train(num_epochs, device, lr, **kwargs)
        pbar = tqdm(range(num_epochs), total=num_epochs)
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])

        train = Flowers102("data/", split="train", download=True, transform=transform)
        val   = Flowers102("data/", split="val",   transform=transform)
        test  = Flowers102("data/", split="test",  transform=transform)

        dataset = ConcatDataset([train, val, test])
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4,drop_last=True)
        batch_size = 64
        # with profile(
        #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True
        # ) as prof:
        step = 0
        for epoch in pbar:
            for z, y in dataloader:
                z = z.to(device)
                y = y.to(device)
               
                self.opt.zero_grad()
                probs = torch.full((batch_size,), self.eta)

                masks = torch.bernoulli(probs).to(self.device)
                y = torch.where(masks==1, y, 10).to(z.device)
                t = torch.rand(batch_size,1,1,1).to(z.device)

                # x = self.path.sample_conditional_path(z, t)
                eps = torch.randn_like(z)
                zt = t * z + (1 - t) * eps
                x_pred = self.model(zt, t, y)
                loss = torch.mean(((x_pred - z) ** 2), dim=(1,2,3)).mean()                  
                # loss = torch.nn.functional.mse_loss(self.model(x, t, y), self.path.conditional_vector_field(x, z, t)) / batch_size
                loss.backward()
                self.opt.step()
                self.scheduler.step()
                pbar.set_description(f'Step {step}, loss: {loss.item():.8f}')
                step += 1
                
            # prof.step()
            if epoch % 10 == 0:
                os.makedirs('results/flower_jit/', exist_ok=True)
                torch.save(self.model.state_dict(), f'results/flower_jit/{self.model.__class__.__name__}_epoch_{epoch}.pt')
                # Play with these!
                samples_per_class = 1
                num_timesteps = 500
                guidance_scales = [1.0, 3.0, 5.0]

                # Graph
                fig, axes = plt.subplots(1, len(guidance_scales), figsize=(10 * len(guidance_scales), 10))
                for idx, w in enumerate(guidance_scales):
                    # Setup ode and simulator
                    ode = CFGXODE(self.model, guidance_scale=w)
                    simulator = EulerSimulator(ode)

                    # Sample initial conditions
                    # y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.int64).repeat_interleave(samples_per_class).to(device)
                    y = torch.randint(0, 102, (1,), dtype=torch.int64).to(device).repeat_interleave(samples_per_class).to(device)
                    
                    num_samples = y.shape[0]
                    x0, _ = self.path.p_simple.sample(num_samples) # (num_samples, 1, 32, 32)
                    # Simulate
                    ts = torch.linspace(0,1,num_timesteps).view(1, -1, 1, 1, 1).expand(num_samples, -1, 1, 1, 1).to(device)
                    x1 = simulator.simulate(x0, ts, y=y)

                    # Plot
                    x1 = x1.detach().cpu()
                    grid = make_grid(x1, nrow=samples_per_class, normalize=True, value_range=(-1,1))
                    axes[idx].imshow(grid.permute(1, 2, 0).cpu())
                    axes[idx].axis("off")
                    axes[idx].set_title(f"Guidance: $w={w:.1f}$", fontsize=25)
                plt.savefig(f'results/flower_jit/{self.model.__class__.__name__}_epoch_{epoch}.png')
                plt.close(fig)
