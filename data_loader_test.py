"""
Benchmark the data loading performance of PyTorch's DataLoader with different num_workers settings.
Expected Output: Prints the time taken to load 100 batches for each worker configuration (0, 2, 4, 8).
"""

import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def benchmark_dataloader(num_workers):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Increase dataset size here
    dataset = datasets.FakeData(size=100000, transform=transform)

    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True)

    start = time.time()
    for i, (data, target) in enumerate(loader):
        if i >= 100:
            break
    end = time.time()

    print(f"num_workers={num_workers}: Time taken for 100 batches: {end - start:.2f} seconds")

if __name__ == "__main__":
    for nw in [0, 2, 4, 8]:
        benchmark_dataloader(nw)