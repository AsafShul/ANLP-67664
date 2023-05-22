# imports:
import os
import sys

import torch

import wandb
import numpy as np
from evaluate import load
from datasets import load_dataset
from transformers import (
    EvalPrediction,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# init output path:
RUNNING_ON_COLAB = '/content' == os.getcwd()
BASE_PATH = '/content/gdrive/MyDrive/aml' if RUNNING_ON_COLAB else '.'
print(f'Running on colab: {RUNNING_ON_COLAB}')


def format_path(dir_path):
    return os.path.join(BASE_PATH, dir_path)


def get_config():
    """
    Get the config of the experiment
    :return: seeds, num_train, num_val, num_test
    """
    run_args = dict(zip(['num_seeds', 'num_train', 'num_val', 'num_test'], np.array(sys.argv[1:5], dtype=int)))
    run_args['seeds'] = np.arange(run_args['num_seeds'])
    training_args = TrainingArguments(output_dir=format_path("results"), evaluation_strategy="epoch", save_strategy="epoch")

    return run_args, training_args


def get_run_name(run_args, model_name, seed):
    """
    Get the name of the run
    :param run_args: the run args
    :param model_name: the name of the model
    :param seed: the seed
    :return: the name of the run
    """
    return f'{model_name}_seed-{seed}trainSamples-{run_args["num_train"]}_valSamples-{run_args["num_val"]}_testSamples-{run_args["num_test"]}'


def init_dirs():
    """
    Initialize the folders of the project
    :return:
    """
    work_dirs = ["models", "data", "results"]
    for dir_ in work_dirs:
        dir_path = format_path(dir_)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Folder '{dir_path}' created.")
        else:
            print(f"Folder '{dir_path}' already exists.")


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
        result = tokenizer(examples['sentence'], truncation=True)
        return result

    return preprocess_function


def get_datasets(run_args, seed, tokenizer):
    pre_process_func = get_tokenizer_func(tokenizer)
    raw_datasets = load_dataset('sst2', cache_dir="./data")

    train_dataset = raw_datasets['train'].map(pre_process_func, batched=True, batch_size=None)
    eval_dataset = raw_datasets['validation'].map(pre_process_func, batched=True, batch_size=None)

    if run_args["num_train"] != -1:
        train_dataset = train_dataset.shuffle(seed=seed).select(range(run_args["num_train"]))
    if run_args["num_val"] != -1:
        eval_dataset = eval_dataset.shuffle(seed=seed).select(range(run_args["num_val"]))

    return train_dataset, eval_dataset


def get_test_dataset(num_test):
    test_dataset = load_dataset('sst2', cache_dir="./data", split='test')

    if num_test != -1:
        test_dataset = test_dataset.select(range(num_test))

    return test_dataset


def login_wandb(LOG_WITH_WANDB):
    """
    Login to wandb
    :param LOG_WITH_WANDB: bool - whether to log with wandb
    :return: None
    """
    if not LOG_WITH_WANDB:
        return

    wandb.login()


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

    wandb.init(project="ANLP-ex1_asafshul_f", config=config, name=name)


def finish_wandb(LOG_WITH_WANDB):
    """
    Finish wandb
    :param LOG_WITH_WANDB: bool - whether to log with wandb
    :return: None
    """
    if not LOG_WITH_WANDB:
        return

    wandb.finish()


def filter_results(res, run_args, models):
    filtered_res = {}
    training_time, eval_time = [], []
    best_model_plus_seed_name = None

    best_acc = -np.inf
    for model_name in models:
        model_acc = []
        for seed in run_args['seeds']:
            acc = res[f'{model_name}_{seed}']['eval']['eval_accuracy']
            if acc > best_acc:
                best_acc = acc
                best_model_plus_seed_name = f'{model_name}_{seed}'

            model_acc.append(acc)
            training_time.append(res[f'{model_name}_{seed}']['train'].metrics['train_runtime'])
            eval_time.append(res[f'{model_name}_{seed}']['eval']['eval_runtime'])

        model_acc = np.array(model_acc)
        filtered_res[model_name] = {'eval_acc_mean': model_acc.mean(),
                                    'eval_acc_std': model_acc.std()}
    return filtered_res, training_time, eval_time, best_model_plus_seed_name


def save_results_to_file(filtered_res, training_time, eval_time, models):
    with open(format_path('res.txt'), 'w') as f:
        for model_name in models:
            f.write(
                f'{model_name},{filtered_res[model_name]["eval_acc_mean"]} +- {filtered_res[model_name]["eval_acc_std"]}\n')
        f.write('----\n')
        f.write(f'train time,{np.array(training_time).sum()}\n')
        f.write(f'predict time,{np.array(eval_time).sum()}\n')


def save_test_predictions_to_file(best_model_plus_seed_name, num_test):
    # load best model and predict:
    model_name = best_model_plus_seed_name.split('_')[0]
    model = AutoModelForSequenceClassification.from_pretrained(format_path(f'models/{best_model_plus_seed_name}'))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_dataset = get_test_dataset(num_test)

    model.eval()
    preds = []
    with torch.no_grad():
        for sample in test_dataset:
            tokenized_sample = tokenizer(sample['sentence'], truncation=True, return_tensors='pt')
            preds.append(model(**tokenized_sample).logits.argmax(dim=1).cpu().numpy()[0])

    print(preds)
    with open(format_path('predictions.txt'), 'w') as f:
        for sentence, label in zip(test_dataset['sentence'], preds):
            f.write(f'{sentence}###{label}\n')
