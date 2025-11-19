from .base import *
from torch import nn
import torch
from torchvision.datasets import Flowers102
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
class IsotropicGaussian(nn.Module, Sampleable):
    """
    Sampleable wrapper around torch.randn
    """
    def __init__(self, shape: List[int], std: float = 1.0):
        """
        shape: shape of sampled data
        """
        super().__init__()
        self.shape = shape
        self.std = std
        self.dummy = nn.Buffer(torch.zeros(1)) # Will automatically be moved when self.to(...) is called...
        # self.dummy = torch.zeros(1).to("cuda")
        
    def sample(self, num_samples) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.std * torch.randn(num_samples, *self.shape).to(self.dummy.device), None

class CifraSampler(nn.Module, Sampleable):
    """
    Sampleable wrapper for the Cifra-10 dataset
    """
    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.dataset = datasets.CIFAR10(
            root='./data/',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])
        )
        print(f"dataset size: {len(self.dataset)}")
        self.dummy = torch.zeros(1).to(self.device) # Will automatically be moved when self.to(...) is called...

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, c, h, w)
            - labels: shape (batch_size, label_dim)
        """
        if num_samples > len(self.dataset):
            raise ValueError(f"num_samples exceeds dataset size: {len(self.dataset)}")
        
        indices = torch.randperm(len(self.dataset))[:num_samples]
        samples, labels = zip(*[self.dataset[i] for i in indices])
        samples = torch.stack(samples).to(self.dummy)
        labels = torch.tensor(labels, dtype=torch.int64).to(self.dummy.device)
        return samples, labels
    
class FlowerSampler(nn.Module, Sampleable):
    """
    Sampleable wrapper for the Flowers-102 dataset
    """
    def __init__(self, device: str):
        super().__init__()
        self.device = device
        
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])

        train = Flowers102("data/", split="train", download=True, transform=transform)
        val   = Flowers102("data/", split="val",   transform=transform)
        test  = Flowers102("data/", split="test",  transform=transform)

        self.dataset = ConcatDataset([train, val, test])
        # self.dataset = ConcatDataset([train])
        
        print(f"dataset size: {len(self.dataset)}")
        self.dummy = torch.zeros(1).to(self.device) # Will automatically be moved when self.to(...) is called...
        self.dataloader = None
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, c, h, w)
            - labels: shape (batch_size, label_dim)
        """
        if self.dataloader is None:
            self.dataloader = DataLoader(self.dataset, batch_size=num_samples, shuffle=True, num_workers=4,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=4)
        # if num_samples > len(self.dataset):
        #     raise ValueError(f"num_samples exceeds dataset size: {len(self.dataset)}")
        
        # indices = torch.randperm(len(self.dataset))[:num_samples]
        # samples, labels = zip(*[self.dataset[i] for i in indices])
        # samples = torch.stack(samples).to(self.dummy)
        # labels = torch.tensor(labels, dtype=torch.int64).to(self.dummy.device)
        # return samples, labels
        samples, labels = next(iter(self.dataloader))
        samples = samples.to(self.dummy.device)
        labels = labels.to(self.dummy.device)
        return samples, labels
