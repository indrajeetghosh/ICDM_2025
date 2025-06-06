import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class WoMDataset(Dataset):
 
    def __init__(self, frames, hmaps, augment=True):
        self.augment = augment
        self.frames, self.hmaps = frames, hmaps

        self.color_jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        )

        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = self.frames[idx]          
        hm  = self.hmaps [idx]         
        
        if self.augment:
            if random.random() < 0.5:    # Horizontal flip
                img = np.fliplr(img).copy()
                hm  = np.fliplr(hm ).copy()

        if self.augment:
            img = self.color_jitter(TF.to_pil_image(img))

        img = self.to_tensor_norm(img)
        hm  = torch.from_numpy(hm).unsqueeze(0).float() / 255.0

        return img, hm
