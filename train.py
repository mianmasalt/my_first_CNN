import model
import torch
import dataset
from torch import nn
import matplotlib.pyplot as plt
import figure

def train(epochs,train_Loader,val_Loader,model,criterion,optimizer,device):
    train_loss_list=[]
    train_acc_list=[]
    val_loss_list=[]
    val_acc_list=[]

    for epoch in range(epochs):
        model.train()
       
        train_loss=0
        train_acc=0
        train_total=0
        best_val_acc=0
        for batch_idx,(images,labels) in enumerate(train_Loader):
            images=images.to(device)
            labels=labels.to(device)

            optimizer.zero_grad()
            outputs=model.foward(images)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            train_total+=labels.size(0)
            predicted=torch.argmax(outputs,dim=1)
            train_acc+=(predicted==labels).sum().item()
            train_loss+=loss.item()

        train_acc/=train_total
        train_loss/=len(train_Loader) #train_total是样本数量，len(dataset.train_Loader)是批次数量
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        model.eval()
        val_acc=0
        val_total=0
        val_loss=0
        with torch.no_grad():
            model.eval()
            for images,labels in val_Loader:
                images=images.to(device)
                labels=labels.to(device)

                outputs=model.foward(images)
                loss=criterion(outputs,labels)

                val_total+=labels.size(0)
                predicted=torch.argmax(outputs,dim=1)
                val_acc+=(predicted==labels).sum().item()
                val_loss+=loss.item()
                if val_acc>best_val_acc:
                    best_val_acc=val_acc
                    torch.save(model.state_dict(),"best_model.pth")

        val_acc/=val_total
        val_loss/=len(val_Loader)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss:{train_loss:.4f},Train Accuracy:{train_acc:.4f} | "
              f"Val Loss:{val_loss:.4f},Val Accuracy:{val_acc:.4f}")

    figure.CNN_plot(range(1,epochs+1),[train_loss_list,val_loss_list],["Train","Validation"],"Train and Validation Loss","Epochs","Loss","train_loss.png")
    figure.CNN_plot(range(1,epochs+1),[train_acc_list,val_acc_list],["Train","Validation"],"Train and Validation Accuracy","Epochs","Accuracy","train_accuracy.png")

device=torch.device("cuda")
model=model.CNN().to(device)

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

epochs=10
train_Loader=dataset.get_data_loaders("train")
val_Loader=dataset.get_data_loaders("val")
train(epochs, train_Loader, val_Loader, model, criterion, optimizer, device)