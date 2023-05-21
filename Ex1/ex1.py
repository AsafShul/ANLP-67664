# imports:
import wandb
import torch
import numpy as np


from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    set_seed,
)

import utils

# constants:
LOG_WITH_WANDB = True
MODELS = ['bert-base-uncased', 'roberta-base', 'google/electra-base-generator']

if __name__ == '__main__':
    # init runtime:
    utils.init_dirs()
    run_args, training_args = utils.get_config()

    # load data:
    res = {}

    for seed in run_args['seeds']:
        set_seed(seed)
        for model_name in MODELS:
            utils.init_wandb(LOG_WITH_WANDB, config=training_args, name=utils.get_run_name(run_args, model_name, seed))
            config = AutoConfig.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, cache_dir="./data")
            train_dataset, eval_dataset = utils.get_datasets(run_args, seed, tokenizer)

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=utils.get_metric_func('accuracy'),
                tokenizer=tokenizer,
            )
            train_result = trainer.train()

            model.save_pretrained(f'./models/{model_name}_{seed}')
            model.eval()
            eval_results = trainer.evaluate(eval_dataset=eval_dataset)

            res[f'{model_name}_{seed}'] = {'train': train_result,
                                           'eval': eval_results,
                                           'model': model,
                                           'tokenizer': tokenizer,
                                           'model_name': model_name,
                                           'seed': seed}
            if LOG_WITH_WANDB:
                wandb.finish()

    print(res)

    filtered_res = {}
    training_time, eval_time = [], []
    best_model_plus_seed_name = None
    best_acc = -np.inf
    for model_name in MODELS:
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
    print('-' * 50)
    print(filtered_res)

    # write results to res.txt in the format
    with open('res.txt', 'w') as f:
        for model_name in MODELS:
            f.write(
                f'{model_name},{filtered_res[model_name]["eval_acc_mean"]} +- {filtered_res[model_name]["eval_acc_std"]}\n')
        f.write('----\n')
        f.write(f'train time,{np.array(training_time).sum()}\n')
        f.write(f'predict time,{np.array(eval_time).sum()}\n')

    # load best model and predict:
    seed = res[best_model_plus_seed_name]['seed']
    model_name = res[best_model_plus_seed_name]['model_name']
    model = AutoModelForSequenceClassification.from_pretrained(f'./models/{model_name}_{seed}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_dataset = utils.get_test_dataset(run_args, seed, tokenizer)

    test_inputs = tokenizer(test_dataset['sentence'], truncation=True, padding=True, return_tensors='pt')

    # Generate predictions
    model.to('cpu')
    model.eval()
    with torch.no_grad():
        outputs = model(**test_inputs)
        # outputs = model(test_dataset)
        predicted_labels = outputs.logits.argmax(dim=1)

    with open('predictions.txt', 'w') as f:
        for sentence, label in zip(test_dataset['sentence'], predicted_labels):
            f.write(f'{sentence}###{label}\n')
