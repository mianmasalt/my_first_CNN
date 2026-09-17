import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.classifier=nn.Sequential(nn.Flatten(),
                                      nn.Linear(in_features=64*16*16,out_features=128),
                                      nn.ReLU(),
                                      nn.Linear(128,2))

    def foward(self,x):
        y=self.features(x)
        y=self.classifier(y)
        return y