from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import models
import glob
import os
import Factory

def train(**kwargs):
    config = Factory.get_config(kwargs)
    dataset = Factory.get_dataset(kwargs)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                            num_workers=config.num_workers, pin_memory=True)

    model = getattr(models, config.model)(config)
    model.setup(config)

    # for epoch in range(config.epochs, )

def predict(**kwargs):
    pass



if __name__ == '__main__':
    import fire
    fire.Fire()