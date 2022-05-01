import numpy as np
import os
import argparse
from utils.argparser_helper import reduce_to_single_arguments, check_args_positive_numbers, print_arguments
import matplotlib.pyplot as plt
import json
from results_common.results_against_time_single_tests import get_tests as get_resuls_against_time_single_tests
from results_common.results_single_states import do_single_states

def do_average_states(output_dir, params):
    pass


def get_arguments():
    parser = argparse.ArgumentParser(description="Tests performance and speed of numerical solution.")
    parser.add_argument('--GRID_SIZE', type=int, nargs='?', default=100,
                        help='Grid size of the model to evaluate.')
    parser.add_argument('--TRUTH_GRID_SIZE', type=int, nargs='?', default=0,
                        help='(Optional) For potentials with no analytical solution, the size of the grid to use to classify as a the true solution. Set to zero to use analytical models only.')
    parser.add_argument('--NUM_TIME_INTERVALS', type=int, nargs='?', default=20,
                        help='(Optional) The number of time intervals to sample over. Defaults to 20.')
    parser.add_argument('--MAX_TIME', type=float, nargs='?', default=1.2,
                        help='(Optional) The maximum time to use when evolving states. Defaults to 1.2.')
    parser.add_argument('--ITERATIONS', type=int, nargs='?', default=5,
                        help='(Optional) The number of iterations to take for each timestep. Defaults to 5.')                  

    args = vars(parser.parse_args())
    reduce_to_single_arguments(args)
    check_arguments(args)
    print_arguments(args)
    return args


def check_arguments(args):
    if args['TRUTH_GRID_SIZE'] < 0:
        raise ValueError('\'TRUTH_GRID_SIZE\' argument must not be less than zero.')

    if args['TRUTH_GRID_SIZE'] != 0:
        raise NotImplementedError('Truth for large grid size not supported yet.')

    check_args_positive_numbers(args, ['GRID_SIZE', 'NUM_TIME_INTERVALS', 'MAX_TIME'])


def get_output_folder():
    if not os.path.isdir('performance'):
        print('Making directory \'performance\'.')
        os.mkdir('performance')

    if not os.path.isdir(os.path.join('performance', 'results_against_time_evolve_traditional')):
        print(f'Making directory \'{os.path.join("performance", "results_against_time_evolve_traditional")}\'.')
        os.mkdir(os.path.join('performance', 'results_against_time_evolve_traditional'))

    num = 1
    while True:
        directory = f'performance/results_against_time_evolve_traditional/{num}'
        if not os.path.exists(directory):
            print(f'Making directory \'{directory}\'.')
            os.mkdir(directory)
            return directory
        num += 1


def create_params_file(args, directory):
    with open(f'{directory}/params.json', 'w') as params_file:
        params_file.write(json.dumps(args))


def main():
    params = get_arguments()

    params['SIMULATION_TYPE'] = 'NUMERICAL'

    # Get output directory
    output_directory = get_output_folder()
    create_params_file(params, output_directory)

    do_single_states(output_directory, params)
    do_averaged_states(output_directory, params)
    

if __name__ == "__main__":
    main()
