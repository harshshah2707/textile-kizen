import cv2
import os
import glob
from pathlib import Path

# Paths
IMAGE_DIR = "saved_frames"
LABEL_DIR = "datasets/custom_real/labels"
IMAGE_OUT_DIR = "datasets/custom_real/images"

# Ensure directories exist
Path(LABEL_DIR).mkdir(parents=True, exist_ok=True)
Path(IMAGE_OUT_DIR).mkdir(parents=True, exist_ok=True)

class Annotator:
    def __init__(self):
        self.images = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
        self.current_idx = 0
        self.boxes = []
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.img = None
        self.temp_img = None

    def draw_box(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.temp_img = self.img.copy()
                cv2.rectangle(self.temp_img, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            cv2.rectangle(self.img, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            # YOLO Format: [class, x_center, y_center, width, height] (normalized)
            h, w = self.img.shape[:2]
            xc = (self.ix + x) / 2 / w
            yc = (self.iy + y) / 2 / h
            bw = abs(x - self.ix) / w
            bh = abs(y - self.iy) / h
            self.boxes.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    def run(self):
        if not self.images:
            print("No images found in saved_frames/!")
            return

        cv2.namedWindow("Annotator")
        cv2.setMouseCallback("Annotator", self.draw_box)

        while self.current_idx < len(self.images):
            img_path = self.images[self.current_idx]
            self.img = cv2.imread(img_path)
            self.temp_img = self.img.copy()
            self.boxes = []
            
            print(f"Annotating ({self.current_idx+1}/{len(self.images)}): {img_path}")
            print("  - Draw boxes around defects")
            print("  - Press [S] to Save and Next")
            print("  - Press [R] to Reset current image")
            print("  - Press [Q] to Quit")

            while True:
                display = self.temp_img if self.drawing else self.img
                cv2.imshow("Annotator", display)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s'):
                    # Save image and label
                    img_name = Path(img_path).name
                    cv2.imwrite(os.path.join(IMAGE_OUT_DIR, img_name), cv2.imread(img_path))
                    with open(os.path.join(LABEL_DIR, img_name.replace(".jpg", ".txt")), "w") as f:
                        f.write("\n".join(self.boxes))
                    self.current_idx += 1
                    break
                elif key == ord('r'):
                    self.img = cv2.imread(img_path)
                    self.temp_img = self.img.copy()
                    self.boxes = []
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    return

        cv2.destroyAllWindows()
        print("\nAnnotation complete! Data saved to datasets/custom_real/")
        print("Now run: python finetune_custom.py")

if __name__ == "__main__":
    Annotator().run()
