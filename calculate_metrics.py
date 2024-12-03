import argparse
import os
import numpy
import numpy as np
import torch
from torch import tensor
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.regression import MeanSquaredError

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


class _PeakSignalNoiseRatio:
    def __init__(self, data_range=1.0):
        self.data_range = data_range

    def __call__(self, pred, target):
        # 确保输入是PyTorch张量
        if not isinstance(pred, torch.Tensor):
            pred = torch.tensor(pred, dtype=torch.float32)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.float32)

        # 计算MSE
        mse = torch.mean((pred - target) ** 2)

        # 计算PSNR
        psnr = 20 * torch.log10(1.0 / mse)
        return psnr.item()

class Metrics:
    def __init__(self, path, idx=None):
        # self.psnr = _PeakSignalNoiseRatio(data_range=2.0)
        # self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        # self.mse = MeanSquaredError()

        self.metric_names = ['psnr', 'ssim', 'mse']
        self.file_path = os.path.join(path, "npy")

        # 存路径
        self.realA = []
        self.realB = []
        self.fakeB = []

        self.realA_name = []
        self.realB_name = []
        self.fakeB_name = []

        for file in os.listdir(self.file_path):
            p = file[:-4]
            if idx is not None and file[:-15].endswith(str(idx)) is False:
                continue

            if p.endswith("A"):
                self.realA.append(os.path.join(self.file_path, file))
                # self.targetsA.append(torch.tensor(np.load(os.path.join(self.file_path, file))))
                self.realA_name.append(file)
            elif p.endswith("B"):
                pp = p[:-2]
                if pp.endswith("real"):
                    self.realB.append(os.path.join(self.file_path, file))
                    # self.targetsB.append(torch.tensor(np.load(os.path.join(self.file_path, file))))
                    self.realB_name.append(file)
                else:
                    self.fakeB.append(os.path.join(self.file_path, file))
                    # self.preds.append(torch.tensor(np.load(os.path.join(self.file_path, file))))
                    self.fakeB_name.append(file)

    # 计算PSNR
    def calculate_psnr(self, img1, img2):
        return psnr(img1, img2, data_range=img1.max() - img1.min())

    # 计算SSIM
    def calculate_ssim(self, img1, img2):
        return ssim(img1, img2, data_range=img1.max() - img1.min())

    # 计算MSE
    def calculate_mse(self, img1, img2):
        return torch.mean((torch.tensor(img1, dtype=torch.float) - torch.tensor(img2, dtype=torch.float)) ** 2)

    def test(self):
        for i in range(0, len(self.fakeB_name)):
            print("{}\t{}\t{}\n".format(self.fakeB_name[i], self.realA_name[i], self.realB_name[i]))

    # def compute(self):
    #     psnr = 0.0
    #     ssim = 0.0
    #     mse = 0.0
    #     num = len(self.preds)
    #     for i in range(0, num):
    #         psnr += self.psnr(self.preds[i], self.targetsA[i])
    #         ssim += self.ssim(self.preds[i].unsqueeze(0).unsqueeze(0), self.targetsA[i].unsqueeze(0).unsqueeze(0))
    #         mse += self.mse(self.preds[i], self.targetsA[i])
    #
    #     psnr /= num
    #     ssim /= num
    #     mse /= num
    #
    #     return psnr, ssim, mse

    def compute2(self):
        psnrs = []
        ssims = []
        mses = []
        num = len(self.fakeB)
        for i in range(0, num):
            realA_img = np.load(self.realA[i])
            realB_img = np.load(self.realB[i])
            fakeB_img = np.load(self.fakeB[i])
            psnr_value = self.calculate_psnr(realB_img, fakeB_img)
            # psnr_value = self.psnr(torch.tensor(realB_img), torch.tensor(fakeB_img))
            ssim_value = self.calculate_ssim(realB_img, fakeB_img)
            mse_value = self.calculate_mse(realB_img, fakeB_img)

            psnrs.append(psnr_value)
            ssims.append(ssim_value)
            mses.append(mse_value)

        return psnrs, ssims, mses

    def calculate_average_metrics(self, psnrs, ssims, mses):
        """
            Input: psnrs, ssims, mses的list
        """
        psnr_mean, psnr_std = np.mean(psnrs), np.std(psnrs)
        ssim_mean, ssim_std = np.mean(ssims), np.std(ssims)
        mse_mean, mse_std = np.mean(mses), np.std(mses)
        return (psnr_mean, psnr_std), (ssim_mean, ssim_std), (mse_mean, mse_std)        # 返回均值和标准差



def show_metrics(**kwargs):
    file_path = kwargs['file_path']
    idx = kwargs['idx']
    # file_path = "output/HisGAN_Baseline_Histloss_t1_to_t2"
    # file_path = "output/Pix2pixModel_t1_to_t2"
    # file_path = "output/ResCycleGANModel_t1_to_t2"
    # file_path = "output/HisGAN_Baseline_t1_to_t2"
    metrics = Metrics(file_path, idx)
    # metrics.test()
    psnrs, ssims, mses = metrics.compute2()
    psnr, ssim, mse = metrics.calculate_average_metrics(psnrs, ssims, mses)

    # 输出结果
    print(f"PSNR - Mean: {psnr[0]}, Std: {psnr[1]}")
    print(f"SSIM - Mean: {ssim[0]}, Std: {ssim[1]}")
    print(f"MSE - Mean: {mse[0]}, Std: {mse[1]}")


if __name__ == '__main__':
    import fire
    fire.Fire()

'''
python -u calculate_metrics.py show_metrics --file_path='output/monet2photo_CycleGANModel_trainA_to_trainB'

python -u calculate_metrics.py show_metrics --file_path='output/Pix2pixModel_t1_to_t2' --idx 82
python -u calculate_metrics.py show_metrics --file_path='output/ResCycleGANModel_t1_to_t2' --idx 82
python -u calculate_metrics.py show_metrics --file_path='output/CycleGANModel1_t1_to_t2' --idx 82
python -u calculate_metrics.py show_metrics --file_path='output/MultiCycleGANModel_t1_to_t2' --idx 82
python -u calculate_metrics.py show_metrics --file_path='output/MedGANModel_t1_to_t2' --idx 82
python -u calculate_metrics.py show_metrics --file_path='output/ARGANModel_t1_to_t2' --idx 82


python -u calculate_metrics.py show_metrics --file_path='output/HisGAN_Baseline_t1_to_t2' --idx 82
python -u calculate_metrics.py show_metrics --file_path='output/HisGAN_Baseline_Histloss_t1_to_t2' --idx 82
python -u calculate_metrics.py show_metrics --file_path='output/HisGAN_EMANet_t1_to_t2' --idx 82
python -u calculate_metrics.py show_metrics --file_path='output/HisGAN_EMANet_Histloss_t1_to_t2' --idx 82


python -u calculate_metrics.py show_metrics --file_path='output/ResCycleGANModel_flair_to_t1ce' --idx 82

'''

