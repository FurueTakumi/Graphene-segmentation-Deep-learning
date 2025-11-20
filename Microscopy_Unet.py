import sys
import os
import random
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sklearn.metrics as metrics

from unet import UNet


class Segmentation_Network:
    def __init__(
        self,
        D,
        W,
        reset_optim=False,
        model_name="UNET",
        LR=0.0001,
        load=None,
        save_freq=10,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.D = D
        self.W = W
        self.load = load
        self.LR = LR
        self.save_freq = save_freq
        self.e = 0
        self.i = 0
        self.init_time = time.time()
        self.reset_optim = reset_optim
        self.model_name = model_name
        self.save_path = self.model_name
        self.data = []  # データ未読込み時の初期値

        self.Model_Init()

    # ------------------- model & io -------------------

    def Model_Init(self):
        print("\ninitializing model")
        self.model = UNet(
            n_classes=4,         # ★ 4クラス
            padding=True,
            up_mode="upconv",
            depth=self.D,
            wf=self.W,
        ).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), self.LR)

        if self.load is None:
            print(">>> from blank slate")
        else:
            print(">>> from saved model")
            self.load_model()

        # CrossEntropyLoss: 入力 (N, C, H, W), 教師 (N, H, W) int64
        self.criterion = nn.CrossEntropyLoss()

    def save_model(self):
        print(" --> saving model")
        torch.save(
            {
                "epoch": self.e,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
                "loss": float(getattr(self, "loss", 0.0)),
            },
            self.save_path + ".pth",
        )

    def load_model(self):
        print("loading model from", self.load)
        checkpoint = torch.load(self.load, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.e = 0
        self.l = 0
        if (not self.reset_optim) and ("optimizer_state_dict" in checkpoint):
            self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
            self.e = checkpoint.get("epoch", 0)
            self.l = checkpoint.get("loss", 0)
        print("reseting optimizer")

    # ------------------- dataset -------------------

    def GetDataSet(self, img_path, label_path, handle="_LABEL"):
        """
        img_path   : 元のTEM画像フォルダ (training_images/)
        label_path : 0/1/2/3 に変換済みラベルフォルダ (new_train_labels_int/)
        handle     : ラベルファイル名に付いている "_LABEL" など
        """
        files = os.listdir(label_path)
        all_data = []
        self.data_handle = handle

        for file in files:
            if self.data_handle in file:
                # 例: name_LABEL.png -> prefix="name", suffix=".png"
                prefix, suffix = file.split(self.data_handle)
                data_fp = os.path.join(img_path, prefix + suffix)      # 入力画像
                label_fp = os.path.join(label_path, file)              # ラベル画像

                d = cv2.imread(data_fp, cv2.IMREAD_GRAYSCALE)
                l = cv2.imread(label_fp, cv2.IMREAD_UNCHANGED)

                if d is None or l is None:
                    continue

                # ★ ラベルは 0,1,2,3 の整数マスクをそのまま使う
                if l.ndim == 3:
                    # もし誤ってカラーで保存されていたら1chに落とす
                    l = l[:, :, 0]
                l = l.astype(np.int64)

                all_data.append((d, l, file))

        if not all_data:
            raise RuntimeError("No training data found. Check paths / handle name.")
        self.data = all_data
        print(f"[INFO] Loaded {len(self.data)} image/label pairs.")

    def DataShuffle(self, seed):
        random.seed(seed)
        random.shuffle(self.data)

    # ------------------- tracking -------------------

    def init_progress_track(self):
        # ★ 4クラス分の Dice/F1 を保存
        self.metrics_df = pd.DataFrame(
            columns=["F1_0", "F1_1", "F1_2", "F1_3", "epoch", "time"]
        )

    def update_progress_track(self):
        f1_0, f1_1, f1_2, f1_3 = self.Validate()
        row = {
            "F1_0": f1_0,
            "F1_1": f1_1,
            "F1_2": f1_2,
            "F1_3": f1_3,
            "epoch": self.e,
            "time": time.time() - self.init_time,
        }
        self.metrics_df = pd.concat(
            [self.metrics_df, pd.DataFrame([row])], ignore_index=True
        )

    def save_progress_track(self):
        self.metrics_df.to_csv(self.save_path + ".csv", index=False)

    # ------------------- helpers -------------------

    @staticmethod
    def _safe_standardize(x, eps=1e-8):
        mu = np.mean(x)
        std = np.std(x)
        if std < eps:
            return x - mu
        return (x - mu) / (std + eps)

    @staticmethod
    def _to_int8(img):
        img = img - np.min(img)
        m = np.max(img)
        if m > 0:
            img = img / m
        img = (img * 255).astype(np.uint8)
        return img

    def PreProcess(self, x):
        return self._safe_standardize(x)

    # ------------------- train / eval -------------------

    def train(self, epochs=300):
        for e in range(epochs):
            self.e = e
            self.update_progress_track()
            self.save_progress_track()
            random.shuffle(self.training_data)
            for i in range(len(self.training_data)):
                self.i = i
                self.iterate()
            if e % self.save_freq == 0:
                self.save_model()

    def Kfold(self, k=5, epochs=300):
        if not self.data:
            raise RuntimeError("No dataset loaded. Call GetDataSet(...) first.")
        print(
            f"\n========== RUNNING {k}-FOLD CROSS VALIDATION ==========\n",
            end="",
        )

        indices = np.array_split(np.arange(len(self.data)), k)

        for fold, test_idx in enumerate(indices):
            print(
                f"\n ############### STARTING FOLD: {fold} of {k} ###############"
            )
            self.Model_Init()
            self.init_progress_track()
            self.save_path = f"{self.model_name}_fold_{fold}"

            test_idx = test_idx.tolist()
            train_idx = [i for i in range(len(self.data)) if i not in test_idx]

            self.test_data = [self.data[i] for i in test_idx]
            self.training_data = [self.data[i] for i in train_idx]

            self.train(epochs)

    def iterate(self):
        x, gt, f = self.training_data[self.i]
        x = self.PreProcess(x)

        x = np.reshape(x, [1, 1, x.shape[0], x.shape[1]])  # (N=1,C=1,H,W)
        gt = np.reshape(gt, [1, gt.shape[0], gt.shape[1]])  # (N=1,H,W)

        x = torch.from_numpy(x).float().to(self.device)
        gt = torch.from_numpy(gt).long().to(self.device)

        self.optim.zero_grad()
        y = self.model(x)           # (1,4,H,W)
        loss = self.criterion(y, gt)
        loss.backward()
        self.optim.step()
        self.loss = loss

        text = (
            f"\repoch:{self.e}\t\tmodel:{self.save_path}"
            f"\t\titeration:{self.i}\t\tloss:{loss.item():.5f}"
        )
        sys.stdout.write(text)
        sys.stdout.flush()

    def Validate(self):
        print("\nEvaluating... ", end="")
        l, w = (self.test_data[0][0]).shape
        z = len(self.test_data)

        y_pred = np.zeros([l, w, z], dtype=np.int64)
        y_true = np.zeros([l, w, z], dtype=np.int64)
        images = []

        for idx, (x, gt, name) in enumerate(self.test_data):
            prediction = self.infer(x)
            y_pred[:, :, idx] = prediction
            y_true[:, :, idx] = gt
            images.append((x, prediction, gt))

        progr_folder = self.save_path + "_ep/"
        if not os.path.exists(progr_folder):
            os.makedirs(progr_folder)

        check = self.stich(images)
        cv2.imwrite(os.path.join(progr_folder, f"{self.e}.png"), check)

        y_pred_r = y_pred.ravel()
        y_true_r = y_true.ravel()
        # ★ 4クラス分のF1（=Dice）を計算
        f1s = metrics.f1_score(
            y_true=y_true_r,
            y_pred=y_pred_r,
            labels=[0, 1, 2, 3],
            average=None,
        )
        f1_0, f1_1, f1_2, f1_3 = [float(v) for v in f1s]
        print(
            f"Dice class0(graphene): {f1_0:.3f}  "
            f"class1(vacuum): {f1_1:.3f}  "
            f"class2(amorph.): {f1_2:.3f}  "
            f"class3(GB): {f1_3:.3f}"
        )
        return f1_0, f1_1, f1_2, f1_3

    def infer(self, x):
        x = self.PreProcess(x)
        x = np.reshape(x, [1, 1, x.shape[0], x.shape[1]])
        x = torch.from_numpy(x).float().to(self.device)
        x = torch.cat((x, x), 0)  # 元コード準拠（N=2にして1枚目だけ使用）
        y_out = self.model(x)
        y_out = y_out[0].squeeze().detach().cpu()  # (4,H,W)
        prediction = torch.argmax(y_out, dim=0).numpy()  # (H,W)
        return prediction

    def stich(self, imgs):
        x, p, gt = imgs[0]
        x = self._to_int8(x)
        gt = self._to_int8(gt)
        p = self._to_int8(p)
        stitch = np.concatenate((x, p, gt), axis=1)
        for x, p, gt in imgs[1:]:
            x = self._to_int8(x)
            gt = self._to_int8(gt)
            p = self._to_int8(p)
            add = np.concatenate((x, p, gt), axis=1)
            stitch = np.concatenate((stitch, add), axis=0)
        return stitch
