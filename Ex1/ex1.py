# imports:
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
    utils.login_wandb(LOG_WITH_WANDB)
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

            trainer.save_model(utils.format_path(f'models/{model_name}_{seed}'))

            model.eval()
            eval_results = trainer.evaluate(eval_dataset=eval_dataset)

            res[f'{model_name}_{seed}'] = {'train': train_result,
                                           'eval': eval_results,
                                           'model': model,
                                           'tokenizer': tokenizer,
                                           'model_name': model_name,
                                           'seed': seed}

            utils.finish_wandb(LOG_WITH_WANDB)

    filtered_res, training_time, eval_time, best_model_plus_seed_name = utils.filter_results(res, run_args, MODELS)
    utils.save_results_to_file(filtered_res, training_time, eval_time, MODELS)
    utils.save_test_predictions_to_file(best_model_plus_seed_name, run_args['num_test'])

