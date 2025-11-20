import os
import glob
import cv2
import numpy as np

# ===== 設定 =====
# GIMPで作ったカラーラベルPNGを置くフォルダ
INPUT_DIR = "new_train_labels_color"

# 0/1/2/3 の整数ラベルを書き出すフォルダ
OUTPUT_DIR = "new_train_labels_int"

# OpenCV(BGR)での色 → クラス番号の対応
# 0: graphene, 1: grain boundary, 2: vacuum, 3: contaminant
COLOR2LABEL = {
    (0,   0,   0): 0,  # black = graphene
    (0,   0, 255): 1,  # red   = grain boundary
    (255, 0,   0): 2,  # blue  = vacuum
    (0, 255,   0): 3,  # green = contaminant
}

def ensure_dir(path: str):
    """フォルダがなければ作成"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[INFO] create dir: {path}")

def print_unique_colors(img: np.ndarray, fname: str):
    """
    RGB色のユニーク値を表示（デバッグ用）
    """
    # img: (H, W, 3), BGR
    h, w, c = img.shape
    assert c == 3, f"Expected 3 channels, got {c} for {fname}"
    unique = np.unique(img.reshape(-1, 3), axis=0)
    print(f"\n[INFO] Unique BGR colors in {fname}:")
    for b, g, r in unique:
        print(f"  BGR=({b:3d}, {g:3d}, {r:3d})")

def convert_label_img(img_bgr: np.ndarray, fname: str) -> np.ndarray:
    """
    BGRカラーラベル画像 → 0/1/2/3 の整数ラベル画像に変換
    """
    h, w, _ = img_bgr.shape
    label = np.zeros((h, w), dtype=np.uint8)

    # まず全部 0 (graphene) にしておく
    # → GIMPの透明部分が (0,0,0) になるので、そのままClass 0扱いでOK

    # 定義した色に対応する部分だけ上書き
    used_colors = set()

    for bgr, cls in COLOR2LABEL.items():
        mask = np.all(img_bgr == np.array(bgr, dtype=np.uint8), axis=-1)
        if np.any(mask):
            label[mask] = cls
            used_colors.add(bgr)

    # 予期しない色が混ざっていないかチェック
    unique_after = np.unique(img_bgr.reshape(-1, 3), axis=0)
    defined_colors = set(COLOR2LABEL.keys())
    unexpected = [tuple(c) for c in unique_after if tuple(c) not in defined_colors]

    # 黒(0,0,0) は Graphene として扱うので、COLOR2LABELに入っている
    # その他の色がある場合はワーニング
    if len(unexpected) > 0:
        print(f"[WARN] {fname}: unexpected colors found (not mapped to any class):")
        for c in unexpected:
            print(f"       BGR={c}")
        print("       → アンチエイリアスや半透明が残っていないか確認してください。")

    return label

def main():
    ensure_dir(INPUT_DIR)
    ensure_dir(OUTPUT_DIR)

    png_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))

    if not png_files:
        print(f"[INFO] No PNG files found in {INPUT_DIR}.")
        print("       GIMPで作成した *_LABEL.png をここに入れてください。")
        return

    print(f"[INFO] Found {len(png_files)} label image(s) in {INPUT_DIR}.")

    for fp in png_files:
        fname = os.path.basename(fp)
        print(f"\n[INFO] Processing: {fname}")

        # BGRで読み込み（アルファチャンネルは無視）
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[ERROR] Failed to read {fp}")
            continue

        # まずユニークな色を表示（デバッグ用）
        print_unique_colors(img, fname)

        # 色マスク → 0/1/2/3 ラベルに変換
        label = convert_label_img(img, fname)

        # 出力ファイル名：元の名前をそのまま使う（任意で _int を付けてもOK）
        out_fp = os.path.join(OUTPUT_DIR, fname)
        cv2.imwrite(out_fp, label)
        print(f"[INFO] Saved label map to: {out_fp}")

    print("\n[DONE] All label images have been converted.")

if __name__ == "__main__":
    main()
