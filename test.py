import torch
import model
import dataset


def test(model,test_Loader,device):
    model.eval()
    test_acc=0
    test_total=0

    with torch.no_grad():
        for images,labels in test_Loader:
            images=images.to(device)
            labels=labels.to(device)

            outputs=model.foward(images)
            predicted=torch.argmax(outputs,dim=1)

            test_total+=labels.size(0)
            test_acc+=(predicted==labels).sum().item()

    print(f"Test Accuracy: {test_acc/test_total:.4f}")

device=torch.device("cuda")
model=model.CNN().to(device)
model.load_state_dict(torch.load("best_model.pth"))
test_Loader=dataset.get_data_loaders("test")

test(model,test_Loader,device)
