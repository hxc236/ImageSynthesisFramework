from torch.utils.data import DataLoader
import models
from config.monet2photo_config import Monet2PhotoConfig
from data.monet2photo_dataset import Monet2PhotoDataset
from train_utils import get_config, get_dataset

def train(**kwargs):
    print('kwargs:', kwargs)

    config = get_config(kwargs)
    dataset = get_dataset(config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                            num_workers=6, pin_memory=True)

    model = getattr(models, config.model)(config)
    model.setup(config)

    # for epoch in range(config.epochs, )

def predict(**kwargs):
    pass



if __name__ == '__main__':
    import fire
    fire.Fire()