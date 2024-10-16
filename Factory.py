import config
from config.monet2photo_config import Monet2PhotoConfig

import data



def get_config(kwargs):
    global config
    dataset = kwargs.dataset
    if dataset == 'monet2photo':
        config = Monet2PhotoConfig()
        config._parse(**kwargs)

    return config



def get_dataset(kwargs):
    dataset = kwargs.dataset
    if dataset == 'monet2photo':
        dataset = Monet2PhotoDataset(config)
    else:
        raise NotImplementedError('Dataset [{}] is not recognized.'.format(dataset))