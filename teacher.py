# train_teacher.py

import torch
import torch.nn.functional as F
from tqdm import tqdm


def loss_teacher(pred_score, gt_score, kind="smooth_l1"):
    
    if kind == "l1":
        return F.l1_loss(pred_score, gt_score)
    elif kind == "smooth_l1":
        return F.smooth_l1_loss(pred_score, gt_score)
    elif kind == "mse":
        return F.mse_loss(pred_score, gt_score)
    else:
        raise ValueError(f"Unknown loss kind: {kind}")


# ----- Teacher training loop -----
def train_teacher_epoch(teacher, loader, opt, device):

    teacher.train()
    running = 0.0

    for imgs, scores, _ in tqdm(loader, desc="Teacher-train", leave=True):
        imgs   = imgs.to(device)
        scores = scores.to(device).view(-1, 1)

        _, t_score = teacher(imgs)

 
        loss = loss_teacher(t_score, scores, kind="mse") #or l1, smooth_li

        opt.zero_grad()
        loss.backward()
        opt.step()

        running += loss.item()

    avg_loss = running / len(loader)
    return avg_loss
    

# ----- Teacher evaluation -----
@torch.no_grad()
def eval_teacher(teacher, loader, device):
    
    teacher.eval()
    running = 0.0

    for imgs, scores, _ in loader:
        imgs   = imgs.to(device)
        scores = scores.to(device).view(-1, 1)

        _, t_score = teacher(imgs)

        running += F.mse_loss(t_score, scores).item()

    avg_mse = running / len(loader)
    return avg_mse
