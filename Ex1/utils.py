# imports:
import os
import wandb
import argparse
import numpy as np


# util functions:
def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('num_seeds', help='number of seeds to be used for each model', type=int)
    parser.add_argument('num_train', help='number of samples to be used during training', type=int)
    parser.add_argument('num_val', help='number of samples to be used during validation', type=int)
    parser.add_argument('num_test', help='number of samples for which the model will predict a sentiment', type=int)
    args = parser.parse_args()

    return np.arange(args.num_seeds), args.num_train, args.num_val, args.num_test


def init_wandb(LOG_WITH_WANDB, config='default', name='default'):
    if not LOG_WITH_WANDB:
        return
    wandb.login(key=os.environ['WANDB_API_KEY'], relogin=True)
    wandb.init(project="ANLP-ex1", config=config, name=name)