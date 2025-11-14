import hyperspy.api as hs
import cv2, numpy as np, os
from pathlib import Path

# --- 設定 ---
in_path  = Path("./inference_images/sample.dm4")  # 入力DM4
out_dir  = Path("./inference_tiles"); out_dir.mkdir(parents=True, exist_ok=True)
tile     = 256
stride   = 192
rb_sigma = 40

# === DM4読込 ===
s = hs.load(str(in_path))
img = s.data.astype(np.float32)

# === 照明むら補正 ===
bg = cv2.GaussianBlur(img, (0,0), rb_sigma)
img_corr = img - bg

# === 標準化 ===
mu, sd = float(img_corr.mean()), float(img_corr.std() + 1e-8)
img_norm = (img_corr - mu) / sd

# === タイル分割 ===
H, W = img_norm.shape[:2]
def tiles_with_overlap(arr, size=256, stride=192):
    for y in range(0, H - size + 1, stride):
        for x in range(0, W - size + 1, stride):
            yield x, y, arr[y:y+size, x:x+size]

win1d = np.hanning(tile); win2d = np.sqrt(np.outer(win1d, win1d)).astype(np.float32)

for x, y, patch in tiles_with_overlap(img_norm, tile, stride):
    patch_u8 = cv2.normalize(patch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(str(out_dir / f"tile_x{x}_y{y}.tif"), patch_u8)
    np.savez_compressed(out_dir / f"weight_x{x}_y{y}.npz", w=win2d)

print("✅ 前処理完了：", len(list(out_dir.glob('*.tif'))), "枚のタイルを出力しました。")
