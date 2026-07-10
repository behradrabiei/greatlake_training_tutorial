"""Minimal single-node DDP training demo on MNIST."""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

EPOCHS = 2
BATCH_SIZE = 64
NUM_WORKERS = 2
DATA_DIR = "./data"


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    loss_tensor = torch.tensor([total_loss, total_samples], device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    return loss_tensor[0].item() / loss_tensor[1].item()


def main():
    local_rank = setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(
            f"Starting DDP training: world_size={world_size}, "
            f"rank={rank}, local_rank={local_rank}"
        )

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = DDP(SimpleCNN().to(device), device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        avg_loss = train_epoch(model, loader, optimizer, device)
        if rank == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {avg_loss:.4f}")

    if rank == 0:
        print("Training complete.")

    cleanup()


if __name__ == "__main__":
    main()
