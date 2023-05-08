# imports:
import wandb
import numpy as np
import pandas as pd

import utils

# constants:
LOG_WITH_WANDB = False
MODELS = ['bert-base-uncased', 'roberta-base', 'google/electra-base-generator']


def batch_end_callback(trainer):
    if trainer.iter_num % 10 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
        if LOG_WITH_WANDB:
            wandb.log({"train_loss": trainer.loss.item()})


if __name__ == '__main__':
    # init runtime:
    utils.init_wandb(LOG_WITH_WANDB)  # todo - add config and name
    seeds, num_train, num_val, num_test = utils.get_config()

    res = pd.DataFrame(columns=['model', 'seed', 'acc'])
    for model in MODELS:
        best_seed = None
        for seed in seeds:
            pass  # fine-tune model
            res.at[len(res)] = [model, seed, acc]

    # save results to res.txt file:
    with open('res.txt', 'w') as f:
        for model, model_df in res.groupby('model'):
            f.write(f'{model},{model_df["acc"].mean():.2f} +- {model_df["acc"].std():.2f}\n')

