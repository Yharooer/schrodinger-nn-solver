import scipy.interpolate
import numpy as np
import argparse
import os
from utils.argparser_helper import reduce_to_single_arguments, check_args_positive_numbers, print_arguments
from utils.dataset import SchrodingerDataset
import json
import pickle
import time
import torch

def get_arguments():
    parser = argparse.ArgumentParser(description="Creates training data.")
    parser.add_argument('TIME_INTERVAL', type=float, nargs=1,
                        help='The maximum time a state will be evolved over.')
    parser.add_argument('--VALIDATION_TIME_INTERVAL', type=float, nargs='?', default=-1,
                        help='The maximum time a state will be evolved over for validation.')
    parser.add_argument('--NUM_TIME_STEPS', type=int, nargs='?', default=50,
                        help='[Optional] Evaluate the solution psi(t) at this many points in time. Defaults to 50.')
    parser.add_argument('--NUM_INITIAL_STATES', type=int, nargs='?', default=200,
                        help='[Optional] Number of initial states to evolve over time. Defaults to 200.')
    parser.add_argument('--NUM_FOURIER_MODES', type=int, nargs='?', default=8,
                        help='[Optional] Number of Fourier modes to be used to generate the initial state. Defaults to 8.')
    parser.add_argument('--NUM_POTENTIAL_DEGREE', type=int, nargs='?', default=8,
                        help='[Optional] Number of the highest degree polynomial to be used to generate the potential. Defaults to 8. Set to -1 to turn off potential.')
    parser.add_argument('--SIMULATION_GRID_SIZE', type=int, nargs='?', default=100,
                        help='[Optional] Size of the grid to do numerical simulations on. Defaults to 100.')
    parser.add_argument('--TRAINING_GRID_SIZE', type=int, nargs='?', default=100,
                        help='[Optional] Size of the grid which the wavefunction is sampled to generate the training data. Defaults to 100. This really should be equal to or less than the simulation grid size.')
    parser.add_argument('--RANDOM_X_SAMPLING', action='store_true',
                        help='Add this argument to sample wavefunction at random positions.')
    parser.add_argument('--RANDOM_T_SAMPLING', action='store_true',
                        help='Add this argument to sample wavefunction at points in time.')
    parser.add_argument('--BATCH_TIME_EVAL_SIZE', type=int, nargs='?', default=1000,
                        help='Number of times to evaluate in one go. Only for use with RANDOM_T_SAMPLING. Defaults to 1000.')
    parser.add_argument('--POTENTIAL_SCALE_FACTOR', type=float, nargs='?', default=0,
                        help='Will scale the potential by a random number sampled with mean 1 standard deviation UNSUPERVISED_POTENTIAL_SCALING. A value greater than one will have the tendancy of making the potentials larger. If zero is provided, will not scale.')
    parser.add_argument('--UNSUPERVISED', action='store_true',
                        help='Add this argument and will only generate initial states. Will generate empty target.')

    args = vars(parser.parse_args())
    reduce_to_single_arguments(args, ['TIME_INTERVAL'])
    check_arguments(args) 
    print_arguments(args)
    return args


def check_arguments(args):
    if args['VALIDATION_TIME_INTERVAL'] < 0:
        args['VALIDATION_TIME_INTERVAL'] = args['TIME_INTERVAL']

    check_args_positive_numbers(args, ['TIME_INTERVAL', 'NUM_TIME_STEPS', 'NUM_INITIAL_STATES', 'NUM_FOURIER_MODES', 'TRAINING_GRID_SIZE', 'SIMULATION_GRID_SIZE', 'VALIDATION_TIME_INTERVAL', 'BATCH_TIME_EVAL_SIZE'])


def get_output_folder():
    if not os.path.isdir('training_data'):
        print('Making directory \'training_data\'.')
        os.mkdir('training_data')

    num = 1
    while True:
        directory = f'training_data/{num}'
        if not os.path.exists(directory):
            print(f'Making directory \'{directory}\'')
            os.mkdir(directory)
            return directory
        num += 1


def create_params_file(args, directory):
    with open(f'{directory}/params.json', 'w') as params_file:
        params_file.write(json.dumps(args))


def main():
    params = get_arguments()

    # Create output directory
    output_dir = get_output_folder()
    create_params_file(params, output_dir)

    # Generate training data.
    dataset = SchrodingerDataset(
        simulation_grid_size=params['SIMULATION_GRID_SIZE'],
        training_grid_size=params['TRAINING_GRID_SIZE'],
        fourier_modes=params['NUM_FOURIER_MODES'],
        potential_degree=params['NUM_POTENTIAL_DEGREE'],
        max_time=params['TIME_INTERVAL'],
        ntimes=params['NUM_TIME_STEPS'],
        num_initials=params['NUM_INITIAL_STATES'],
        random_x_sampling=params['RANDOM_X_SAMPLING'],
        random_t_sampling=params['RANDOM_T_SAMPLING'],
        potential_scale_factor=params['POTENTIAL_SCALE_FACTOR'],
        batch_time_eval_size=params['BATCH_TIME_EVAL_SIZE'],
        unsupervised=params['UNSUPERVISED']
    )

    print('Generating training data. This may take a while...')
    start = time.perf_counter()
    dataset.initialise()
    end = time.perf_counter()
    print('Finished generating training data!\n')

    # Generate validation data
    print('Generating validation data. This may take a while...')
    validation_dataset = SchrodingerDataset(
        simulation_grid_size=params['SIMULATION_GRID_SIZE'],
        training_grid_size=100,
        fourier_modes=params['NUM_FOURIER_MODES'],
        potential_degree=params['NUM_POTENTIAL_DEGREE'],
        max_time=params['VALIDATION_TIME_INTERVAL'],
        ntimes=10,
        num_initials=50,
        random_x_sampling=False,
        random_t_sampling=False,
        potential_scale_factor=params['POTENTIAL_SCALE_FACTOR'],
        unsupervised=False
    )
    validation_dataset.initialise()
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=1000, shuffle=True)
    validation_x, validation_y = next(iter(validation_dataloader))
    print('Finished generating validation data!\n')

    # Update params with final time
    params['TIME_TAKEN_SECONDS'] = end-start
    create_params_file(params, output_dir)

    # Save dataset.
    # This takes a huge amount of time oh my gosh
    print('Saving datasets...')
    with open(f'{output_dir}/training_dataset.pt', 'wb') as file:
        torch.save(dataset.get_state_dict(), file)
    print(f'Training Dataset successfully saved to \'{output_dir}/training_dataset.pt\'.')

    with open(f'{output_dir}/validation_dataset.pt', 'wb') as file:
        torch.save({
            'validation_x': validation_x,
            'validation_y': validation_y
        }, file)
    print(f'Validation Dataset successfully saved to \'{output_dir}/validation_dataset.pt\'.')


if __name__ == "__main__":
    main()
