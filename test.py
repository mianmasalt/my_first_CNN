import torch
import model
import dataset
from torchvision.models import (resnet18,ResNet18_Weights)
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import confusion_matrix,auc,roc_auc_score,roc_curve,precision_recall_curve,average_precision_score
import figure
import numpy as np
import pandas as pd
def test(model,test_Loader,device,threshold,pth):
    checkpoint_path = Path(__file__).resolve().parent / "checkpoints" / pth
    checkpoint=torch.load(checkpoint_path,map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    test_acc=0
    test_total=0
    true_labels=[]
    predicted_labels=[]
    all_probabilities=[]
    with torch.no_grad():
        for images,labels in test_Loader:
            
            images=images.to(device)
            labels=labels.to(device)

            outputs=model.forward(images)
            probabilities=torch.softmax(outputs,dim=1)
            all_probabilities.extend(probabilities[:,1].cpu().numpy())

            predicted=(probabilities[:,1]>=threshold).long()
           
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

            test_total+=labels.size(0)
            test_acc+=(predicted==labels).sum().item()

    cm=confusion_matrix(
        true_labels,
        predicted_labels
    )

    print("The threshold is ",threshold)

    print(cm)

    TN,FP,FN,TP=cm.ravel()
    
    print(f"TN:{TN}\n",
          f"FP:{FP}\n",
          f"FN:{FN}\n",
          f"TP:{TP}")
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1=2*precision*recall/(precision+recall)

    fpr,tpr,thre=roc_curve(true_labels,all_probabilities)
    roc_auc=auc(fpr,tpr)


    P,R,thre=precision_recall_curve(true_labels,all_probabilities)
    AP=average_precision_score(true_labels,all_probabilities)

    test_acc=test_acc/test_total
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall is: {recall:.4f}")
    print(f"Test F1 score is : {F1:.4f}\n")

    figure.ROC_plot(fpr,tpr,roc_auc,"the AUC of full FT model ")
    figure.PR_plot(R,P,AP,"the PR of full FT model")

    return {"Accuracy":test_acc,"Precision":precision,"Recall":recall,"F1":F1,"AUC":roc_auc,"AP":AP}



    
device=torch.device("cuda")
model=resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc=nn.Linear(model.fc.in_features,2)


test_Loader=dataset.get_data_loaders("test")

results=[]
metrics=test(model,test_Loader,device,0.5,pth="freeze_model.pth")
metrics["model"]="freeze"
results.append(metrics)

metrics=test(model,test_Loader,device,0.5,pth="best_model.pth")
metrics["model"]="partial"
results.append(metrics)

metrics=test(model,test_Loader,device,0.5,pth="Full_FT_model.pth")
metrics["model"]="full"
results.append(metrics)

df=pd.DataFrame(results)

print(df.to_string(index=False))
#TPR=[]
#FPR=[]
#cur=[]
#for threshold in np.arange(0,1,0.01):
   #cur=test(model,test_Loader,device,threshold)
   #TPR.append(cur[3]/(cur[3]+cur[2]))
   #FPR.append(cur[1]/(cur[1]+cur[0]))

#figure.CNN_plot(FPR,[TPR],"ROC_Curve","ROC","FPR",["TPR"],"ROC_test.png")
