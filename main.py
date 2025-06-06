import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from load_data import load_frame_heatmap_data
from gaze_dataset import GazeDataset
from train_teacher import train_teacher_epoch, eval_teacher
from train_student import train_student_epoch, eval_student
from eval import eval_student
from memgaze_models import TeacherNet, StudentNet


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

csv_file = 'student_kd_eval_metrics.csv'

with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'KD-Train-Loss', 'NSS', 'Spearman'])
    

frame_folder   = 'Images/GT/extracted_frames'
heatmap_folder = 'Images/Gaze/extracted_frames'
IMG_SZ = 256
BATCH_SZ = 32
EPOCHS_T = 200
EPOCHS_S = 200
alpha = 0.5
T = 1.5


frames, hmaps = load_frame_heatmap_data(frame_folder, heatmap_folder, IMG_SZ)
print(f"Frames shape: {frames.shape} | Heatmaps shape: {hmaps.shape}")


tr_X, te_X, tr_Y, te_Y = train_test_split(
    frames, hmaps, test_size=0.15, random_state=SEED, shuffle=True
)


train_loader = DataLoader(GazeDataset(tr_X, tr_Y), batch_size=BATCH_SZ, shuffle=True, drop_last=True)
val_loader   = DataLoader(GazeDataset(te_X, te_Y), batch_size=BATCH_SZ, shuffle=False)


for idx, (frames_batch, heatmaps_batch) in enumerate(val_loader):
    print(f"Batch {idx + 1}")
    print(f"Frames shape: {frames_batch.shape}")   
    print(f"Heatmaps shape: {heatmaps_batch.shape}")
    break 


teacher = TeacherNet(pretrained=True).to(device)
student = StudentNet().to(device)

opt_t = torch.optim.Adam(teacher.parameters(), lr=1e-4)
opt_s = torch.optim.Adam(student.parameters(), lr=1e-4)


train_losses, val_losses = [], []

for ep in range(1, EPOCHS_T + 1):
    tr_loss = train_teacher_epoch(teacher, train_loader, opt_t, device)
    val_loss = eval_teacher(teacher, val_loader, device)
    print(f"[Teacher {ep:02d}/{EPOCHS_T}] Train={tr_loss:.4f} | Val={val_loss:.4f}")


for ep in range(1, EPOCHS_S + 1):
    tr_loss_s = train_student_epoch(student, teacher, train_loader, opt_s, device, alpha=alpha, T=T)
    nss_score, spearman_score = eval_student(student, val_loader, device)
    print(f"[Student {ep:02d}/{EPOCHS_S}] KD-Train={tr_loss_s:.4f} | NSS={nss_score:.4f} | Spearman={spearman_score:.4f}")

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([ep, tr_loss_s, nss_score, spearman_score])

torch.save(student.state_dict(), "student_kd.pth")
print("Saved student model to student_kd.pth")