from config.monet2photo_config import Monet2PhotoConfig
from data.monet2photo_dataset import Monet2PhotoDataset


def get_config(kwargs):
    global config
    dataset_name = kwargs['dataset']
    if dataset_name == 'monet2photo':
        config = Monet2PhotoConfig()
    # elif dataset_name == 'BraTS2021':
    #     config = BraTS2021Config()
    else:
        raise NotImplementedError('Dataset [{}] is not recognized.'.format(dataset_name))

    print(config.dataset)
    config._parse(kwargs)
    return config


def get_dataset(config):
    dataset_name = config.dataset
    if dataset_name == 'monet2photo':
        return Monet2PhotoDataset(config)
    # elif dataset_name == 'BraTS2021':
    #     return BraTS2021Dataset(get_config(kwargs))
    else:
        raise NotImplementedError('Dataset [{}] is not recognized.'.format(dataset_name))


