import glob
import os

# from os import pread
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image

'''
    该函数功能如下：
    使用nib.load函数加载NIfTI格式的医学图像文件。
    通过get_fdata方法提取图像数据。
    将提取的数据转换为NumPy数组。
    返回处理后的图像数据数组。
'''
def load_nii_to_array(path):

    image = nib.load(path)
    image = image.get_fdata()
    image = np.array(image)
    return image

def nii2npy(output, idx=82): # 72, 77, 82
    # path = r'D:/Data/BraTS2021_TrainingData/TrainingData'
    path = r'D:/Data/RSNA_ASNR_MICCAI_BraTS2021_ValidationData'
    # path ='/Users/jontysun/Downloads/数据集/BrainT1T2FT/MICCAI_FeTS2021_TrainingData'
    # path ='/Users/jontysun/Downloads/数据集/BrainT1T2FT/MICCAI_FeTS2021_ValidationData'
    '''
        使用glob模块的glob函数根据给定的路径模式(path+'/*')查找所有匹配的文件或文件夹，
        并将它们的路径存储在列表paths中；简而言之，这行代码会获取指定目录下的所有子目录和文件的路径。
        paths其中之一例如：r'D:/Data/RSNA_ASNR_MICCAI_BraTS2021_ValidationData/BraTS2021_00001'
    '''
    paths = glob.glob(path+'/*')
    for p in paths:
        ps = glob.glob(p+'/*.nii.gz')
        # Linux use this:
        # name = p.split('/')[-1]
        # Windows use this:
        name = os.path.basename(p)
        print(name)
        for i in ps:
            image3d = load_nii_to_array(i)  # 获取3d数据
            if 't1.nii' in i:
                tab = 't1'
            elif 't2.nii' in i:
                tab = 't2'
            elif 't1ce.nii' in i:
                tab = 't1ce'
            elif 'flair.nii' in i:
                tab = 'flair'
            else: # seg
                continue 
            image2d = image3d[:, :, idx]
            # plt.imshow(image2d)
            # plt.show()
            # 保存为 xxx/t1/name_idx.npy
            np.save('{}/{}/{}_{}'.format(output, tab, name, idx), image2d)
        # break
        

def test(path):
    image = np.load(path)
    image = Image.fromarray(image)
    image.show()



if __name__ == '__main__':
    # nii2npy(r'D:/PreparedData/BrainT1T2FT/npyFTTT') # train set
    nii2npy(r'D:/PreparedData/BrainT1T2FT/npyFTTTest') # test set npyFTTTest
