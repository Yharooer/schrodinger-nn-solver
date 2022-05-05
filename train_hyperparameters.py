import ray
import ray.tune
from ray.tune.suggest.bayesopt import BayesOptSearch
import os
import matplotlib.pyplot as plt
from train_model import train_model, create_params_file, get_output_folder, extract_training_data_id
from ray.tune.suggest.dragonfly import DragonflySearch
from ray.tune.stopper import TrialPlateauStopper

TRAINING_DATA = 'training_data/38/training_dataset.pt'
VALIDATION_DATA = 'training_data/38/validation_dataset.pt'

NUM_SAMPLES = 400
MAX_EPOCHS = 200
NUM_GPUS = 2
MAX_PER_GPU = 3

training_data_id = extract_training_data_id(TRAINING_DATA)

default_params = {
    'TRAINING_DATA': os.path.abspath(TRAINING_DATA),
    'VALIDATION_DATA': os.path.abspath(VALIDATION_DATA),
    'MAX_EPOCHS': MAX_EPOCHS,
    'BATCH_SIZE': 1000,
    'HIDDEN_LAYER_SIZE': 500,
    'CHECKPOINT_FREQUENCY': 5,
    'FROM_CHECKPOINT': None,
    'USE_AUTOGRAD': False,
    'UNSUPERVISED_RANDOMISED': True,
    'RAND_MAX_TIME': 0.5,
    'TRAINING_MIXING': True,
    'UNSUPERVISED_POTENTIAL_SCALING': 2.0
}

search_space = {
    'LEARNING_RATE': 1e-4,
    'HYPERPARAM_MSE': 0,
    'HYPERPARAM_DT': ray.tune.loguniform(1e-9,1e-3),
    'HYPERPARAM_BC': ray.tune.loguniform(1e-6, 1e0),
    'HYPERPARAM_IC': ray.tune.loguniform(1e-6, 1e0),
    'HYPERPARAM_NORM': 0,
    'HYPERPARAM_ENERGY': 0,
}

def call_at_epoch(epoch, total_losses, component_losses, validation_losses):
    # ray.tune.checkpoint_dir(epoch)

    ray.tune.report(
        epoch=(epoch+1),
        training_loss=total_losses[-1],
        mse_loss=component_losses[0][-1],
        dt_loss=component_losses[1][-1],
        bc_loss=component_losses[2][-1],
        ic_loss=component_losses[3][-1],
        norm_loss=component_losses[4][-1],
        energy_loss=component_losses[5][-1],
        validation_loss=validation_losses[-1]
    )

def training_loop_wrapper(config):
    print(os.getcwd())
    params = {**default_params, **config}
    output_folder = get_output_folder(training_data_id)
    train_model(params, output_folder, call_at_epoch)

def get_hyperparam_output_directory():
    if not os.path.isdir('hyperparam_tune'):
        print('Making directory \'hyperparam_tune\'...')
        os.mkdir('hyperparam_tune')

    num=1
    while True:
        directory = f'hyperparam_tune/{num}'
        if not os.path.exists(directory):
            print(f'Making directory \'{directory}\'.')
            os.mkdir(directory)
            return directory
        num += 1
    
ray.init(
    num_gpus=NUM_GPUS,
    object_store_memory=200 * 1024 * 1024
)

dragonfly_search = DragonflySearch(
    optimizer="bandit",
    domain="euclidean",
    metric='validation_loss',
    mode='min',
    points_to_evaluate=[
        {
            'LEARNING_RATE': 1e-4,
            'HYPERPARAM_MSE': 0,
            'HYPERPARAM_DT': 6e-5,
            'HYPERPARAM_BC': 2e-2,
            'HYPERPARAM_IC': 1e-2,
            'HYPERPARAM_NORM': 0,
            'HYPERPARAM_ENERGY': 0,
        }
    ]
)

trial_stopper = TrialPlateauStopper(
    num_results=10,
    std=0.005,
    metric='validation_loss',
    mode='min'
)

analysis = ray.tune.run(
    training_loop_wrapper,
    metric='validation_loss',
    mode='min',
    config=search_space,
    resources_per_trial= {'cpu': 0 ,'gpu': 1.0/MAX_PER_GPU},
    num_samples=NUM_SAMPLES,
    search_alg=dragonfly_search,
    scheduler=ray.tune.schedulers.ASHAScheduler(max_t=MAX_EPOCHS, grace_period=4),
    stop = trial_stopper
)

# Save results
output_directory = get_hyperparam_output_directory()

dfs = analysis.trial_dataframes
plt.figure(figsize=(7,7))
[d.validation_loss.plot() for d in dfs.values()]
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.yscale('log')
plt.savefig(os.path.join(output_directory, f'validation_losses.pdf'))

analysis.results_df.to_csv(os.path.join(output_directory, f'results_df.csv'))
