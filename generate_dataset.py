import os
import random
import math
import shutil
import yaml
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from tqdm import tqdm
from config import DATASET_DIR, IMAGES_DIR, LABELS_DIR, DATASET_CONFIG, DEFECT_CLASSES, CLASS_NAMES

class TextileDatasetGenerator:
    FABRIC_COLORS = [(230, 220, 210), (200, 195, 185), (180, 175, 165), (160, 155, 145), (210, 200, 180)]

    def __init__(self, config=None):
        self.config = config or DATASET_CONFIG
        self.img_size = self.config["image_size"]
        self.num_images = self.config["num_images"]
        random.seed(self.config["seed"])
        np.random.seed(self.config["seed"])

    def generate_base_texture(self):
        size = self.img_size
        img = Image.new("RGB", (size, size), random.choice(self.FABRIC_COLORS))
        pixels = np.array(img, dtype=np.float32)
        spacing = random.choice([4, 6])
        for y in range(0, size, spacing): pixels[y:y+1, :, :] += np.random.normal(0, 10, (1, size, 3))
        for x in range(0, size, spacing): pixels[:, x:x+1, :] += np.random.normal(0, 10, (size, 1, 3))
        pixels = np.clip(pixels + np.random.normal(0, 3, pixels.shape), 0, 255).astype(np.uint8)
        return Image.fromarray(pixels).filter(ImageFilter.GaussianBlur(radius=0.5))

    def add_hole(self, img, draw):
        w, h = random.randint(20, 60), random.randint(20, 60)
        x, y = random.randint(w, self.img_size-w), random.randint(h, self.img_size-h)
        points = [(x + math.cos(2*math.pi*i/8)*random.uniform(0.6, 1)*w/2, y + math.sin(2*math.pi*i/8)*random.uniform(0.6, 1)*h/2) for i in range(8)]
        draw.polygon(points, fill=(30, 30, 30), outline=(100, 100, 100))
        return 0, x/self.img_size, y/self.img_size, w/self.img_size, h/self.img_size

    def add_burn(self, img, draw):
        w, h = random.randint(30, 80), random.randint(20, 60)
        x, y = random.randint(w, self.img_size-w), random.randint(h, self.img_size-h)
        draw.ellipse([x-w/2, y-h/2, x+w/2, y+h/2], fill=(60, 40, 20))
        draw.ellipse([x-w/4, y-h/4, x+w/4, y+h/4], fill=(30, 20, 10))
        return 1, x/self.img_size, y/self.img_size, w/self.img_size, h/self.img_size

    def add_split_end(self, img, draw):
        x1, y1 = random.randint(50, self.img_size-150), random.randint(50, self.img_size-150)
        length, angle = random.randint(60, 120), random.uniform(0, 2*math.pi)
        x2, y2 = x1 + length*math.cos(angle), y1 + length*math.sin(angle)
        draw.line([(x1, y1), (x2, y2)], fill=(50, 50, 50), width=3)
        return 2, (x1+x2)/2/self.img_size, (y1+y2)/2/self.img_size, abs(x1-x2)/self.img_size+0.02, abs(y1-y2)/self.img_size+0.02

    def add_stain(self, img, draw):
        cx, cy, r = random.randint(60, self.img_size-60), random.randint(60, self.img_size-60), random.randint(20, 50)
        overlay = Image.new("RGBA", (self.img_size, self.img_size), (0,0,0,0))
        ImageDraw.Draw(overlay).ellipse([cx-r, cy-r, cx+r, cy+r], fill=(100, 80, 40, 100))
        img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))
        return 3, cx/self.img_size, cy/self.img_size, 2*r/self.img_size, 2*r/self.img_size

    def add_wrinkle(self, img, draw):
        x, y = random.randint(100, self.img_size-100), random.randint(100, self.img_size-100)
        points = [(x+i*20, y+math.sin(i)*10) for i in range(5)]
        draw.line(points, fill=(180, 180, 180), width=2)
        return 4, (x+40)/self.img_size, y/self.img_size, 100/self.img_size, 40/self.img_size

    def generate_dataset(self):
        for split in ["train", "val"]:
            (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
            (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)
        
        split_idx = int(self.num_images * self.config["train_split"])
        defect_funcs = [self.add_hole, self.add_burn, self.add_split_end, self.add_stain, self.add_wrinkle]
        
        for i in tqdm(range(self.num_images)):
            img = self.generate_base_texture()
            draw, ann = ImageDraw.Draw(img), []
            if random.random() > self.config["normal_ratio"]:
                for _ in range(random.randint(1, 3)):
                    cls, xc, yc, w, h = random.choice(defect_funcs)(img, draw)
                    ann.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            
            split = "train" if i < split_idx else "val"
            img.save(IMAGES_DIR / split / f"textile_{i:04d}.jpg")
            with open(LABELS_DIR / split / f"textile_{i:04d}.txt", "w") as f: f.write("\n".join(ann))

        with open(DATASET_DIR / "dataset.yaml", "w") as f:
            yaml.dump({"path": str(DATASET_DIR), "train": "images/train", "val": "images/val", "names": DEFECT_CLASSES, "nc": len(DEFECT_CLASSES)}, f)

if __name__ == "__main__":
    TextileDatasetGenerator().generate_dataset()
