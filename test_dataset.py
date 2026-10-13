import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import CatDogDataset


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = CatDogDataset(
    root_dir="train_data/cats_dogs/train",
    transform=transform
)

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

print("Dataset size:", len(dataset))
print("Batch count:", len(train_loader))

for batch_idx, (images, labels) in enumerate(train_loader):

    print(f"Batch {batch_idx + 1}")
    print("Images:", images.shape)
    print("Labels:", labels.shape)
    print("Pixel range:", images.min().item(), images.max().item())

    if batch_idx == 2:
        break