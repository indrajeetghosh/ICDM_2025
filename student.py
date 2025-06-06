import torch
import torch.nn.functional as F
from tqdm import tqdm
from pytorch_msssim import ssim  

# ----- KD loss -----
def kd_loss_ssim_kl(student_map, teacher_map, alpha=0.5, T=0.2):
    B = student_map.size(0)
    data_rng = max((teacher_map.max() - teacher_map.min()).item(), 1e-8)

    ssim_loss = 1 - ssim(student_map, teacher_map, data_range=data_rng, size_average=True)

    s_logp = F.log_softmax(student_map.view(B, -1) / T, dim=1)
    t_prob = F.softmax(teacher_map.view(B, -1) / T, dim=1)
    kl_loss = F.kl_div(s_logp, t_prob, reduction="batchmean") * (T ** 2)

    return (1 - alpha) * ssim_loss + alpha * kl_loss

# ----- Student training -----
def train_student_epoch(student, teacher, loader, optimizer, device, alpha=0.5, T=0.75):
    
    student.train()
    teacher.eval()
    total_loss = 0.0

    for frames, _ in tqdm(loader, desc="[Student KD Training]"):
        frames = frames.to(device).float()

        with torch.no_grad():
            t_map, _ = teacher(frames)

        s_map, _ = student(frames)

        loss = kd_loss_ssim_kl(s_map, t_map, alpha=alpha, T=T)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# ----- Student evaluation -----
@torch.no_grad()
def eval_student(student, teacher, loader, device, alpha=0.4, T=1.5):
    student.eval()
    teacher.eval()
    total_loss = 0.0

    for frames, _ in loader:
        frames = frames.to(device).float()

        t_map, _ = teacher(frames)
        s_map, _ = student(frames)

        loss = kd_loss_ssim_kl(s_map, t_map, alpha=alpha, T=T)

        total_loss += loss.item()

    return total_loss / len(loader)
