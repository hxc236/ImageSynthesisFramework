import os

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import models
from config.monet2photo_config import Monet2PhotoConfig
from data.monet2photo_dataset import Monet2PhotoDataset
from models.CycleGAN import CycleGANModel
from train_utils import get_config, get_dataset

def train(**kwargs):
    print('kwargs:', kwargs)

    config = get_config(kwargs)
    dataset = get_dataset(config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                            num_workers=6, pin_memory=True)

    print(dataloader)

    model = getattr(models, config.model)(config)
    model.setup(config)
    # model = CycleGANModel(config)

    for epoch in range(config.epoch_count, config.n_epochs + config.n_epochs_decay + 1):
        model.update_learning_rate()
        if hasattr(model, 'alpha'):
            model['alpha'] = [0.0]
            print('Alpha updated')
        for i, data in enumerate(dataloader):
            model.set_input(data)
            model.optimize_parameters()
            if i % 10 == 0:
                print('Epoch: {}, i: {}, loss: {}'.format(epoch, i, model.get_current_losses()))

        print('saving model')
        model.save_networks('iter_{}'.format(epoch))

    model.save_networks('latest')


def predict(**kwargs):
    print('kwargs: {}'.format(kwargs))

    config = get_config(kwargs)     # 并且根据配置中的数据集名，返回一个数据集对应的配置对象

    config.isTrain = False          # 配置中的是否训练设置为否

    dataset = get_dataset(config)   # 从配置中获取训练集
    print(len(dataset))             # 输出查看数据集的长度

    dataset = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)   # 创建一个数据加载器

    model = getattr(models, config.model)(config)   # 根据配置中的模型名，返回一个模型对象
    model.setup(config)             # 因为isTrain设置为False了，直接在这里加载模型权重，通过load_iter参数设置加载哪一轮的模型
    model.eval()                    # 把model中每个网络都设置为评估模式

    if config.task == 'AtoB':
        task = '{}_to_{}'.format(config.A, config.B)
    else:
        task = '{}_to_{}'.format(config.B, config.A)

    output_path = './output/{}_{}_{}'.format(config.dataset, config.model, task)
    image_path = output_path + '/image'
    npy_path = output_path + '/npy'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)

    for i, data in enumerate(dataset):      # data是__getitem__返回的东西
        i = data['name'][0]     # data['name']是该数据的名称 形如 '00001' 再加个[0]是什么意思？
        # print(i)

        model.set_input(data)       # 根据配置中的task是AtoB还是BtoA，把data中的A、B任务赋值给model的real_A、real_B
        model.test()            # forward计算得到结果
        visuals = model.get_current_visuals()       # 得到一个有序字典，包含model的所有visual_names的值

        # real_A = visuals['real_A'].permute(0, 2, 3, 1)[0, :, :, 0]
        real_A = visuals['real_A'].permute(0, 2, 3, 1)[0, :, :, :3]  # permute(0,2,3,1)把 n*c*h*w → n*h*w*c, c=3
        real_A = real_A.data.detach().numpy()

        np.save(npy_path+'/{}_real_A.npy'.format(i), real_A)
        real_A = (real_A+1)/2.0*255.0
        # real_A = ((real_A-real_A.min())/((real_A.max()-real_A.min())/255))
        real_A = real_A.astype(np.uint8)            # 要想转RGB，得先转成uint8， 如果是转L，就保持float32就行
        image = Image.fromarray(real_A).convert('RGB')
        image.save(image_path+'/{}_real_A.png'.format(i))
        # plt.imshow(real_A, cmap='gray')
        # plt.show()

        fake_B = visuals['fake_B'].permute(0, 2, 3, 1)[0, :, :, :3]
        fake_B = fake_B.data.detach().numpy()
        np.save(npy_path+'/{}_fake_B.npy'.format(i), fake_B)
        fake_B = (fake_B+1)/2.0*255.0
        fake_B = fake_B.astype(np.uint8)
        # fake_B = ((fake_B-fake_B.min())/((fake_B.max()-fake_B.min())/255))
        image = Image.fromarray(fake_B).convert('RGB')
        image.save(image_path+'/{}_fake_B.png'.format(i))
        # plt.imshow(fake_B, cmap='gray')
        # plt.show()

        real_B = visuals['real_B'].permute(0, 2, 3, 1)[0, :, :, :3]
        real_B = real_B.data.detach().numpy()
        np.save(npy_path+'/{}_real_B.npy'.format(i), real_B)
        real_B = (real_B+1)/2.0*255.0
        real_B = real_B.astype(np.uint8)
        # real_B = ((real_B-real_B.min())/((real_B.max()-real_B.min())/255))
        image = Image.fromarray(real_B).convert('RGB')
        image.save(image_path+'/{}_real_B.png'.format(i))
        # plt.imshow(real_B, cmap='gray')


        #####

        # real_A = visuals['random_AB'].permute(0, 2, 3, 1)[0, :, :, 0]
        # real_A = real_A.data.detach().numpy()
        # np.save(npy_path+'/{}_random_AB.npy'.format(i), real_A)
        # real_A = (real_A+1)/2.0*255.0
        # image = Image.fromarray(real_A).convert('L')
        # image.save(image_path+'/{}_random_AB.png'.format(i))




if __name__ == '__main__':
    import fire
    fire.Fire(predict)
    # fire.Fire(train)
    # fire.Fire()

'''
# train
python -u main.py train --dataset monet2photo --model CycleGANModel --batch_size 4

python -u main.py predict --dataset monet2photo --gpu_ids='' --model CycleGANModel --A='trainA' --B='trainB' --load_iter=200 --dataroot='D:\Data\经典风格迁移数据集\monet2photo'
'''