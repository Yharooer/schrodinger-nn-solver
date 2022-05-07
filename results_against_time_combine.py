import numpy as np
import os
import argparse
from utils.argparser_helper import reduce_to_single_arguments, check_args_positive_numbers, print_arguments
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import pandas

def do_combine_data(output_dir, params):
    trad_data = pandas.read_csv(params['TRADITIONAL_DATA'])
    nn_data = pandas.read_csv(params['NN_DATA'])

    mpl.rcParams['font.family'] = 'CMU Bright'
    mpl.rcParams['mathtext.fontset'] = 'cm'

    plt.figure(figsize=(5,5))
    plt.plot(nn_data['ts'], nn_data['mse_error'], color='red')
    plt.plot(trad_data['ts'], trad_data['mse_error'], color='blue')
    plt.legend(['Our Model', 'Numerical Model'], frameon=False)
    plt.xlabel('Model Time')
    plt.ylabel('MSE Error')
    plt.yscale('log')
    plt.title('Our Model vs Numerical Model')
    
    plt.savefig(os.path.join(output_dir, f'combined_errors.pdf'))
    plt.close()

    plt.figure(figsize=(5,5))
    plt.plot(nn_data['ts'], nn_data['eval_time'], color='red')
    plt.plot(trad_data['ts'], trad_data['eval_time'], color='blue')
    plt.yscale('log')
    plt.legend(['Our Model', 'Numerical Model'], frameon=False)
    plt.xlabel('Model Time')
    plt.ylabel('Evaluation Time / s')
    plt.title('Our Model vs Numerical Model')
    
    plt.savefig(os.path.join(output_dir, f'combined_eval_time.pdf'))
    plt.close()


def get_arguments():
    parser = argparse.ArgumentParser(description="Tests performance and speed of numerical solution.")
    parser.add_argument('TRADITIONAL_DATA', type=str, nargs=1,
                        help='CSV simulation data of numerical model.')
    parser.add_argument('NN_DATA', type=str, nargs=1,
                        help='CSV simulation data of neural network model.')
    
    args = vars(parser.parse_args())
    reduce_to_single_arguments(args)
    check_arguments(args)
    print_arguments(args)
    return args


def check_arguments(args):
    if not os.path.isfile(args['TRADITIONAL_DATA']):
        raise ValueError(f'Cannot find csv \'{args["TRADITIONAL_DATA"]}\'.')
    
    if not os.path.isfile(args['NN_DATA']):
        raise ValueError(f'Cannot find csv \'{args["NN_DATA"]}\'.')

    # check_args_positive_numbers(args, [])


def get_output_folder():
    if not os.path.isdir('performance'):
        print('Making directory \'performance\'.')
        os.mkdir('performance')

    if not os.path.isdir(os.path.join('performance', 'results_against_time_evolve_combined')):
        print(f'Making directory \'{os.path.join("performance", "results_against_time_evolve_combined")}\'.')
        os.mkdir(os.path.join('performance', 'results_against_time_evolve_combined'))
    
    num = 1
    while True:
        directory = f'performance/results_against_time_evolve_combined/{num}'
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

    for font in mpl.font_manager.findSystemFonts('fonts'):
        mpl.font_manager.fontManager.addfont(font)

    # Get output directory
    output_directory = get_output_folder()
    create_params_file(params, output_directory)

    do_combine_data(output_directory, params)
    

if __name__ == "__main__":
    main()
