import os
import cv2
import numpy as np
from PIL import Image

def load_image_score_or_heatmap(img_root: str,
                                score_file: str = None,
                                hm_dir: str = None,
                                img_size: int = 256):
    imgs = []
    targets = []
    paths = []


    if score_file is not None:
        print(f'[INFO] Loading images + scores from {img_root} and {score_file}')
        with open(score_file) as f:
            for ln in f:
                img_id, sc = ln.strip().split()
                img_path = os.path.join(img_root, img_id)

                if not os.path.exists(img_path):
                    print(f'[WARN] Missing image: {img_id}')
                    continue

                # Load image as RGB
                img = Image.open(img_path).convert('RGB')
                img = img.resize((img_size, img_size))
                img = np.array(img)

                imgs.append(img)
                targets.append(float(sc))
                paths.append(img_path)


    elif hm_dir is not None:
        print(f'[INFO] Loading images + heatmaps from {img_root} and {hm_dir}')
        frame_list = sorted(os.listdir(img_root))

        for fname in frame_list:
            if fname.startswith('.'):
                continue

            img_path = os.path.join(img_root, fname)
            hm_path = os.path.join(hm_dir, fname)

            img = cv2.imread(img_path)
            hmap = cv2.imread(hm_path, cv2.IMREAD_GRAYSCALE)

            if img is None or hmap is None:
                print(f'[WARN] Could not load {fname}')
                continue

            img = cv2.resize(img, (img_size, img_size))
            hmap = cv2.resize(hmap, (img_size, img_size))
            img = img[..., ::-1]

            imgs.append(img)
            targets.append(hmap)
            paths.append(img_path)

    else:
        raise ValueError('You must provide either score_file or hm_dir.')

    imgs = np.stack(imgs) if imgs else np.empty((0, img_size, img_size, 3))
    targets = np.stack(targets) if targets else np.empty((0,))  # adapt shape automatically

    print(f'[INFO] Loaded {len(imgs)} samples.')

    return imgs, targets, paths
