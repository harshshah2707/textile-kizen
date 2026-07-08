# scratch/process_logo.py
import cv2
import numpy as np
import os

def process():
    img = cv2.imread('images.jpg')
    if img is None:
        print("Error: images.jpg not found")
        return
        
    h, w, c = img.shape
    print(f"Loaded image: {w}x{h}")
    
    # 1. Create transparent logo for dark theme (Kizen Logo Light)
    # Convert white background to transparent
    # We want to change the dark blue/black text to white/light gray
    logo_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Threshold to identify background (white)
    # White is typically R,G,B > 240
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    for y in range(h):
        for x in range(w):
            b, g, r = img[y, x]
            # Check if white background
            if r > 240 and g > 240 and b > 240:
                # Transparent
                logo_rgba[y, x] = [0, 0, 0, 0]
            else:
                # Check if it is the dark text/gear:
                # Let's check if it is part of KIZEN (dark blue/navy: b > r and r < 50)
                # or ENGINEERING (black: r < 50, g < 50, b < 50)
                # Let's change these dark colors to white/silver to be visible on dark background.
                # If R, G, B are all low, it's dark text or gear outline.
                # In the logo, the gear is dark gray/black, KIZEN is navy, ENGINEERING is black.
                # Innovation is our tradition is bright blue.
                
                # Let's check if the color is bright blue (Innovation is our tradition)
                # In BGR: B should be high, R should be lower.
                # Let's check if it's the blue trace or blue text:
                # If it's blue, keep it blue!
                is_blue = (b > 150 and r < 100) or (b > 120 and g > 80 and r < 80)
                
                if is_blue:
                    # Keep original color, fully opaque
                    logo_rgba[y, x] = [b, g, r, 255]
                else:
                    # It's the dark text or gear outline. Make it light silver/white.
                    # Maintain some of the anti-aliasing by scaling the brightness.
                    # The darker the original pixel, the whiter we make it.
                    brightness = 255 - int(gray[y, x])
                    if brightness > 50:
                        logo_rgba[y, x] = [240, 240, 240, 255]
                    else:
                        # Semi-transparent transition pixels
                        logo_rgba[y, x] = [240, 240, 240, int(brightness * 5)]
                        
    # 2. Extract and crop the gear (Left part of the processed logo)
    # The gear is located roughly from x=0 to x=230.
    # Let's crop from x=0 to x=225, y=25 to y=255. Let's find the bounding box of non-transparent pixels in the left area.
    left_side = logo_rgba[:, :230]
    non_transparent = np.where(left_side[:, :, 3] > 0)
    if len(non_transparent[0]) > 0:
        ymin, ymax = np.min(non_transparent[0]), np.max(non_transparent[0])
        xmin, xmax = np.min(non_transparent[1]), np.max(non_transparent[1])
        
        # Add padding
        pad = 5
        ymin = max(0, ymin - pad)
        ymax = min(h, ymax + pad)
        xmin = max(0, xmin - pad)
        xmax = min(230, xmax + pad)
        
        gear_crop = left_side[ymin:ymax, xmin:xmax]
        
        # Save gear
        public_dir = os.path.join("frontend", "public")
        os.makedirs(public_dir, exist_ok=True)
        cv2.imwrite(os.path.join(public_dir, "kizen_gear.png"), gear_crop)
        print("Saved kizen_gear.png")
        
    # Save processed light logo
    cv2.imwrite(os.path.join("frontend", "public", "kizen_logo_light.png"), logo_rgba)
    print("Saved kizen_logo_light.png")

if __name__ == '__main__':
    process()
