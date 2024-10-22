import os

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

    config = get_config(kwargs)

    config.isTrain = False

    dataset = get_dataset(config)
    print(len(dataset))

    dataset = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    model = getattr(models, config.model)(config)
    model.setup(config)
    model.eval()

    if config.task == 'AtoB':
        task = '{}_to_{}'.format(config.A, config.B)
    else:
        task = '{}_to_{}'.format(config.B, config.A)

    output_path = './output/{}_{}'.format(config.model, task)
    image_path = output_path + '/image'
    npy_path = output_path + '/npy'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)

    # ... continue
    # for i, data in enumerate(dataset):
    #     i = data['name'][0]
    #     model.set_input(data)
    #     model.test()
    #     visuals = model.get_current_visuals()
    #
    #     real_A = visuals['real_A'].permute(0, 2, 3, 1)[0, :, :, 0]
    #     real_A = real_A.data.detach().numpy()
    #     np.save(npy_path+'/{}_real_A.npy'.format(i), real_A)
    #     real_A = (real_A+1)/2.0*255.0
    #     # real_A = ((real_A-real_A.min())/((real_A.max()-real_A.min())/255))
    #     image = Image.fromarray(real_A).convert('L')
    #     image.save(image_path+'/{}_real_A.png'.format(i))
    #     # plt.imshow(real_A, cmap='gray')
    #     # plt.show()
    #
    #     fake_B = visuals['fake_B'].permute(0, 2, 3, 1)[0, :, :, 0]
    #     fake_B = fake_B.data.detach().numpy()
    #     np.save(npy_path+'/{}_fake_B.npy'.format(i), fake_B)
    #     fake_B = (fake_B+1)/2.0*255.0
    #     # fake_B = ((fake_B-fake_B.min())/((fake_B.max()-fake_B.min())/255))
    #     image = Image.fromarray(fake_B).convert('L')
    #     image.save(image_path+'/{}_fake_B.png'.format(i))
    #     # plt.imshow(fake_B, cmap='gray')
    #     # plt.show()
    #
    #     real_B = visuals['real_B'].permute(0, 2, 3, 1)[0, :, :, 0]
    #     real_B = real_B.data.detach().numpy()
    #     np.save(npy_path+'/{}_real_B.npy'.format(i), real_B)
    #     real_B = (real_B+1)/2.0*255.0
    #     # real_B = ((real_B-real_B.min())/((real_B.max()-real_B.min())/255))
    #     image = Image.fromarray(real_B).convert('L')
    #     image.save(image_path+'/{}_real_B.png'.format(i))
    #     # plt.imshow(real_B, cmap='gray')
    #     # plt.show()
    #
    #
    #     #####
    #
    #     # real_A = visuals['random_AB'].permute(0, 2, 3, 1)[0, :, :, 0]
    #     # real_A = real_A.data.detach().numpy()
    #     # np.save(npy_path+'/{}_random_AB.npy'.format(i), real_A)
    #     # real_A = (real_A+1)/2.0*255.0
    #     # image = Image.fromarray(real_A).convert('L')
    #     # image.save(image_path+'/{}_random_AB.png'.format(i))




if __name__ == '__main__':
    import fire
    fire.Fire(train)

'''
# train
python -u main.py train --dataset monet2photo --model CycleGANModel --batch_size 4

'''