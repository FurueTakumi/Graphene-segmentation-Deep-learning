import os
import shutil

label_dir = "new_train_labels_int"
image_dir = "training_images"   # 元画像がすでに入っているフォルダ

for fname in os.listdir(label_dir):
    if not fname.lower().endswith((".png", ".tif", ".tiff")):
        continue

    label_src = os.path.join(label_dir, fname)

    # ラベルと同じ名前の元画像がある前提（拡張子だけ同じにしておく）
    base = os.path.splitext(fname)[0]
    img_name = base + ".png"    # ここは実際の拡張子に合わせて変更

    img_src = os.path.join(image_dir, img_name)
    if not os.path.exists(img_src):
        print("[WARN] 元画像がありません:", img_src)
        continue

    # ラベルファイルを training_images にコピー & _LABEL を付けた名前に
    label_dst = os.path.join(image_dir, base + "_LABEL.png")
    shutil.copy2(label_src, label_dst)
    print("paired:", img_src, "<->", label_dst)
