import matplotlib.pyplot as plt
from pathlib import Path



def CNN_plot(X,Y,curve_names,table_name,x_label,y_label,pth):
    plt.figure()
    for i, y in enumerate(Y):
        plt.plot(X, y, label=curve_names[i])
    plt.title(table_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()



    plt.savefig(pth)

    plt.show()
    plt.close()


def ROC_plot(fpr,tpr,auc,pth,filename):
    plt.figure()
    plt.plot(fpr,tpr,label=filename+f" = {auc:.4f}")
    plt.plot([0,1],[0,1],"--",label="Random")
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()


    plt.savefig(pth)
    
    plt.show()
    plt.close()

def PR_plot(recall,precision,AP,pth,filename):
    plt.figure()
    plt.plot(recall,precision,label=filename+f" (AP = {AP:.4f})")

    plt.title("PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()

    plt.savefig(pth)
    
    plt.show()
    plt.close()