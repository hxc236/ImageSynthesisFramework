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
    pass



if __name__ == '__main__':
    import fire
    fire.Fire()