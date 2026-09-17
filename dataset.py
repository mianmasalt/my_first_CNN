import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

def get_data_loaders(state):
    if state=="train":
        transform=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor(),
                              transforms.RandomHorizontalFlip(),
                              transforms.RandomRotation(10),
                              transforms.Normalize(mean=[0.485,0.456,0.406],
                                                   std=[0.229,0.224,0.225])])
        return DataLoader(datasets.ImageFolder("my_CNN/my_first_CNN/train_data/cats_dogs/train",transform=transform),batch_size=32,shuffle=True)
    else:
        transform=transforms.Compose([transforms.Resize((128,128)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485,0.456,0.406],
                                                           std=[0.229,0.224,0.225])])
        if state=="val":
            return DataLoader(datasets.ImageFolder("my_CNN/my_first_CNN/train_data/cats_dogs/val",transform=transform),batch_size=32,shuffle=True)
        else:
            return DataLoader(datasets.ImageFolder("my_CNN/my_first_CNN/train_data/cats_dogs/test",transform=transform),batch_size=32,shuffle=True)
           

   
