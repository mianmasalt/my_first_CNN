import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Dataset
from pathlib import Path
from PIL import Image

class CatDogDataset(Dataset):
    def __init__(self,root_dir,transform=None):
   
        self.root_dir=Path(root_dir)
        self.transform=transform
        self.image_paths=[]

        for path in (self.root_dir / "cat").glob("*"):
            self.image_paths.append((path,0))

        for path in (self.root_dir / "dog").glob("*"):
            self.image_paths.append((path,1))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,index):
        image_path,label=self.image_paths[index]

        image=Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image=self.transform(image)

        return image,label
    
def get_data_loaders(state):
    if state=="train":
        train_transform=transforms.Compose([transforms.RandomResizedCrop(128,scale=(0.8,1)),
                              transforms.RandomHorizontalFlip(),
                              transforms.RandomRotation(10),
                              transforms.ColorJitter(brightness=0.1,contrast=0.1),
                               transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485,0.456,0.406],
                                                   std=[0.229,0.224,0.225]),
                                                  ])
        return DataLoader(datasets.ImageFolder("F:/python_learning/my_CNN/my_first_CNN/train_data/cats_dogs/train",transform=train_transform),batch_size=32,shuffle=True)
    else:
        transform=transforms.Compose([transforms.Resize((128,128)),
                                      transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485,0.456,0.406],
                                                            std=[0.229,0.224,0.225])])
        if state=="val":
            return DataLoader(datasets.ImageFolder("F:/python_learning/my_CNN/my_first_CNN/train_data/cats_dogs/val",transform=transform),batch_size=32,shuffle=True)
        else:
            return DataLoader(datasets.ImageFolder("F:/python_learning/my_CNN/my_first_CNN/train_data/cats_dogs/test",transform=transform),batch_size=32,shuffle=True)
           


