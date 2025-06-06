import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import spearmanr
from pytorch_msssim import ssim  

@torch.no_grad()
def eval_student(student, loader, device):
    student.eval()

    nss_scores = []
    spearman_scores = []

    all_pred_scores = []
    all_gt_scores   = []

    for imgs, gt_hmaps, gt_scores in tqdm(loader, desc="[Student Eval] NSS + Spearman"):
        imgs      = imgs.to(device).float()
        gt_hmaps  = gt_hmaps.to(device).float()
        gt_scores = gt_scores.to(device).view(-1, 1) 

        pred_hmaps, pred_scores = student(imgs)

      
        pred_hmaps = pred_hmaps.squeeze(1).cpu().numpy()  
        gt_hmaps   = gt_hmaps.squeeze(1).cpu().numpy()   
        B = pred_hmaps.shape[0]

        for b in range(B):
            pred = pred_hmaps[b]
            gt   = gt_hmaps[b]

           
            pred_norm = (pred - np.mean(pred)) / (np.std(pred) + 1e-8)
            nss = np.mean(pred_norm * (gt / 255.0))
            nss_scores.append(nss)

        all_pred_scores.extend(pred_scores.squeeze(1).cpu().numpy())
        all_gt_scores.extend(gt_scores.squeeze(1).cpu().numpy())

    rho, _ = spearmanr(all_pred_scores, all_gt_scores)
    if np.isnan(rho):
        rho = 0.0

    avg_NSS      = np.mean(nss_scores)
    avg_Spearman = rho

    return avg_NSS, avg_Spearman
