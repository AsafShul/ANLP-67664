import os
import wandb
import argparse
import numpy as np
import pandas as pd


def _get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('num_seeds', help='number of seeds to be used for each model', type=int)
    parser.add_argument('num_train', help='number of samples to be used during training', type=int)
    parser.add_argument('num_val', help='number of samples to be used during validation', type=int)
    parser.add_argument('num_test', help='number of samples for which the model will predict a sentiment', type=int)
    args = parser.parse_args()

    return args.num_seeds, args.num_train, args.num_val, args.num_test


# def batch_end_callback(trainer):
#     if trainer.iter_num % 10 == 0:
#         print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
#         wandb.log({"train_loss": trainer.loss.item()})


if __name__ == '__main__':
    # wandb.login(key=os.environ['WANDB_API_KEY'], relogin=True)
    # wandb.init(project="ex1", config=train_config, name='gpu_train_colab')
    num_seeds, num_train, num_val, num_test = _get_config()
