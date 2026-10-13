import torch 
from pathlib import Path


def save_bese_model(epoch,model,optimizer,metrics,results,pth):
    torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                **dict(zip(metrics,results))},
               pth)


def save_checkpoint(checkpoint, pth):
    torch.save(checkpoint, pth)

def load_checkpoint(model,optimizer,checkpoint_path,scheduler=None,device="cpu"):
    checkpoint=torch.load(checkpoint_path,map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"]