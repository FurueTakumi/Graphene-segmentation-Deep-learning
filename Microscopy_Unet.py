import sys
import os
import random
import time
import shutil

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as metrics

from unet import UNet


class Segmentation_Network:
    def __init__(self, D, W, reset_optim=False, model_name='UNET',
                 LR=0.0001, load=None, save_freq=10):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        # self.data = self.GetDataSet(data_path)  # 呼び出し側で明示的に実行

    # ------------------- model & io -------------------

    def Model_Init(self):
        print('\ninitializing model')
        self.model = UNet(
            n_classes=2, padding=True, up_mode='upconv',
            depth=self.D, wf=self.W
        ).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), self.LR)

        if self.load is None:
            print('>>>from blank slate')
        else:
            print('>>>from saved model')
            self.load_model()

        self.criterion = nn.CrossEntropyLoss()

    def save_model(self):
        print(' --> saving model')
        torch.save(
            {
                'epoch': self.e,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'loss': float(getattr(self, "loss", 0.0)),
            },
            self.save_path + '.pth'
        )

    def load_model(self):
        print('loading model from', self.load)
        checkpoint = torch.load(self.load, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.e = 0
        self.l = 0
        if not self.reset_optim and 'optimizer_state_dict' in checkpoint:
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            self.e = checkpoint.get('epoch', 0)
            self.l = checkpoint.get('loss', 0)
        print('reseting optimizer')

    # ------------------- dataset -------------------

    def GetDataSet(self, data_path, handle='_LABEL'):
        files = os.listdir(data_path)
        All_Data = []
        self.data_handle = handle
        for file in files:
            if self.data_handle in file:
                prefix, suffix = file.split(self.data_handle)
                data_fp = os.path.join(data_path, prefix + suffix)
                label_fp = os.path.join(data_path, file)

                d = cv2.imread(data_fp, 0)     # grayscale
                l = cv2.imread(label_fp, 0)
                if d is None or l is None:
                    continue
                l = (l > 0).astype(int)
                All_Data.append((d, l, file))
        self.data = All_Data

    def DataShuffle(self, seed):
        random.seed(seed)
        random.shuffle(self.data)

    # ------------------- tracking -------------------

    def init_progress_track(self):
        self.metrics_df = pd.DataFrame(columns=['F1_0', 'F1_1', 'epoch', 'time'])

    def update_progress_track(self):
        f1_0, f1_1 = self.Validate()
        row = {'F1_0': f1_0, 'F1_1': f1_1, 'epoch': self.e,
               'time': time.time() - self.init_time}
        # pandas.append 廃止に対応
        self.metrics_df = pd.concat(
            [self.metrics_df, pd.DataFrame([row])],
            ignore_index=True
        )

    def save_progress_track(self):
        self.metrics_df.to_csv(self.save_path + '.csv', index=False)

    # ------------------- train / eval -------------------

    @staticmethod
    def _safe_standardize(x, eps=1e-8):
        # 分散ゼロ対策（0除算回避）
        mu = np.mean(x)
        std = np.std(x)
        if std < eps:
            return x - mu
        return (x - mu) / (std + eps)

    @staticmethod
    def _to_int8(img):
        # 0除算回避（全0/定数画像対策）
        img = img - np.min(img)
        m = np.max(img)
        if m > 0:
            img = img / m
        img = (img * 255).astype(np.uint8)
        return img

    def PreProcess(self, x):
        return self._safe_standardize(x)

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
        print(f'\n==========RUNNING {k}-FOLD CROSS VALIDATION==========\n', end='')

        # len(data) が k で割り切れなくても安全に分割
        indices = np.array_split(np.arange(len(self.data)), k)

        for fold, test_idx in enumerate(indices):
            print(f"\n ###############STARTING FOLD: {fold} of {k}###############")
            self.Model_Init()
            self.init_progress_track()
            self.save_path = f'{self.model_name}_fold_{fold}'

            test_idx = test_idx.tolist()
            train_idx = [i for i in range(len(self.data)) if i not in test_idx]

            self.test_data = [self.data[i] for i in test_idx]
            self.training_data = [self.data[i] for i in train_idx]

            self.train(epochs)

    def iterate(self):
        x, gt, f = self.training_data[self.i]
        x = self.PreProcess(x)
        x = np.reshape(x, [1, 1, x.shape[0], x.shape[1]])
        gt = np.reshape(gt, [1, gt.shape[0], gt.shape[1]])

        x = torch.from_numpy(x).float().to(self.device)
        gt = torch.from_numpy(gt).long().to(self.device)

        self.optim.zero_grad()
        y = self.model(x)
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
        print('\nEvaluating... ', end='')
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

        progr_folder = self.save_path + '_ep/'
        if not os.path.exists(progr_folder):
            os.makedirs(progr_folder)

        check = self.stich(images)
        cv2.imwrite(os.path.join(progr_folder, f'{self.e}.png'), check)

        y_pred_r = y_pred.ravel()
        y_true_r = y_true.ravel()
        f1_0, f1_1 = metrics.f1_score(
            y_true=y_true_r, y_pred=y_pred_r, labels=[0, 1], average=None
        )
        print(f"DICE class 0: {round(f1_0,3)} DICE class 1: {round(f1_1,3)}")
        return f1_0, f1_1

    def infer(self, x):
        x = self.PreProcess(x)
        x = np.reshape(x, [1, 1, x.shape[0], x.shape[1]])
        x = torch.from_numpy(x).float().to(self.device)
        # 元コード準拠：2サンプルにして推論（片方だけ使用）
        x = torch.cat((x, x), 0)
        y_out = self.model(x)
        y_out = y_out[0].squeeze().detach().cpu()
        prediction = torch.argmax(y_out, dim=0).numpy()
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
