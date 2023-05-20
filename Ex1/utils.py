# imports:
import os
import sys

import wandb
import numpy as np
from datasets import load_dataset
from evaluate import load

from dataclasses import dataclass, field
from transformers import TrainingArguments
from transformers.hf_argparser import HfArgumentParser

from transformers import EvalPrediction


@dataclass
class RunArguments:
    num_seeds: int = field(metadata={'help': 'number of seeds to be used for each model'}, default=1)
    num_train: int = field(metadata={'help': 'number of samples to be used during training'}, default=-1)
    num_val: int = field(metadata={'help': 'number of samples to be used during validation'}, default=-1)
    num_test: int = field(metadata={'help': 'number of samples for which the model will predict a sentiment'},
                          default=-1)


def get_config():
    """
    Get the config of the experiment
    :return: seeds, num_train, num_val, num_test
    """
    sys.argv.append("--output_dir=./results")
    parser = HfArgumentParser((RunArguments, TrainingArguments))
    run_args, training_args = parser.parse_args_into_dataclasses()
    run_args.seeds = np.arange(run_args.num_seeds)
    training_args.evaluation_strategy = "epoch"
    training_args.save_strategy = "epoch"

    return run_args, training_args


def get_run_name(run_args, model_name, seed):
    """
    Get the name of the run
    :param run_args: the run args
    :param model_name: the name of the model
    :param seed: the seed
    :return: the name of the run
    """
    return f'{model_name}_seed-{seed}trainSamples-{run_args.num_train}_valSamples-{run_args.num_val}_testSamples-{run_args.num_test}'


def init_dirs():
    """
    Initialize the folders of the project
    :return:
    """
    work_dirs = ["models", "data", "results"]

    for dir_ in work_dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)
            print(f"Folder '{dir_}' created.")
        else:
            print(f"Folder '{dir_}' already exists.")


def get_metric_func(metric_name):
    metric = load(metric_name)

    def compute_metrics(p: EvalPrediction):
        preds = np.argmax(p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)[metric_name]
        return {metric_name: result}

    return compute_metrics


def get_tokenizer_func(tokenizer):
    def preprocess_function(examples):
        # Tokenize the texts
        result = tokenizer(examples['sentence'], max_length=tokenizer.model_max_length, truncation=True)
        return result

    return preprocess_function


def get_datasets(run_args, seed, tokenizer):
    raw_datasets = load_dataset('sst2', cache_dir="./data")
    tokenized_datasets = raw_datasets.map(get_tokenizer_func(tokenizer), batched=True)

    train_dataset = tokenized_datasets['train']
    eval_dataset = tokenized_datasets['validation']

    if run_args.num_train != -1:
        train_dataset = tokenized_datasets['train'].shuffle(seed=seed).select(range(run_args.num_train))
    if run_args.num_val != -1:
        eval_dataset = tokenized_datasets['validation'].shuffle(seed=seed).select(range(run_args.num_val))

    def preprocess_function(examples):
        # Tokenize the texts
        result = tokenizer(examples['sentence'], max_length=512, truncation=True)
        return result

    train_dataset = train_dataset.map(preprocess_function, batched=True, batch_size=None)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True, batch_size=None)

    return train_dataset, eval_dataset


def get_test_dataset(run_args, seed, tokenizer):
    raw_datasets = load_dataset('sst2', cache_dir="./data")
    raw_datasets.set_format(output_all_columns=True)
    tokenized_datasets = raw_datasets.map(get_tokenizer_func(tokenizer), batched=True, batch_size=100)

    test_dataset = tokenized_datasets['test']

    if run_args.num_test != -1:
        test_dataset = tokenized_datasets['test'].shuffle(seed=seed).select(range(run_args.num_test))

    return test_dataset


def init_wandb(LOG_WITH_WANDB, config='default', name='default'):
    """
    Initialize wandb
    :param LOG_WITH_WANDB: bool - whether to log with wandb
    :param config: the config to be used for wandb
    :param name: the name to be used for wandb
    :return: None
    """
    if not LOG_WITH_WANDB:
        return

    wandb.login(key=os.environ.get('WANDB_API_KEY'), relogin=True)
    wandb.init(project="ANLP-ex1", config=config, name=name)
