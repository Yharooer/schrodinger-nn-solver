import numpy as np
import os
import argparse
from utils.argparser_helper import reduce_to_single_arguments, check_args_positive_numbers, print_arguments
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.interpolate
import json
import pandas

def do_combine_data(output_dir, params):
    trad_data = pandas.read_csv(params['TRADITIONAL_DATA'])
    nn_data = pandas.read_csv(params['NN_DATA'])

    mpl.rcParams['font.family'] = 'EB Garamond'
    mpl.rcParams['font.size'] = 9
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble']="\\usepackage{mathpazo}"

    plt.figure(figsize=(3.4,2.6))
    numline = plt.plot(trad_data['ts'], trad_data['mse_error'], color='#8000ff')
    nnline = plt.plot(nn_data['ts'], nn_data['mse_error'], color='#00b35a')
    plt.xlabel('Model Time')
    plt.ylabel('MSE Error')
    plt.yscale('log')
    plt.title('Comparison of Model Performances', fontsize=9, y=1.25)

    leg = plt.legend([nnline[0], numline[0]], ['Physics-Driven', 'Numerical'], shadow=False, edgecolor='#000000', framealpha=1.0, ncol=3, bbox_to_anchor=(0,1.0,1,0.0), loc='lower left', mode='expand', prop={'size': 8})
    leg.get_frame().set_boxstyle('Square', pad=0.1)
    leg.get_frame().set(capstyle='butt', joinstyle='miter', linewidth=1.0)

    plt.tight_layout()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'combined_errors.pgf'))
    plt.savefig(os.path.join(output_dir, f'combined_errors.pdf'))
    plt.close()

    plt.figure(figsize=(3.4,2.6))
    numline = plt.plot(trad_data['ts'], trad_data['eval_time'], color='#8000ff')
    nnline = plt.plot(nn_data['ts'], nn_data['eval_time'], color='#00b35a')
    plt.yscale('log')
    plt.legend(['Physics-Driven Model', 'Numerical Model'], frameon=False)
    plt.xlabel('Model Time')
    plt.ylabel('Evaluation Time / s')
    plt.title('Comparison of Model Speeds', fontsize=9, y=1.20)

    leg = plt.legend([nnline[0], numline[0]], ['Physics-Driven', 'Numerical'], shadow=False, edgecolor='#000000', framealpha=1.0, ncol=3, bbox_to_anchor=(0,1.0,1,0.0), loc='lower left', mode='expand', prop={'size': 8})
    leg.get_frame().set_boxstyle('Square', pad=0.1)
    leg.get_frame().set(capstyle='butt', joinstyle='miter', linewidth=1.0)

    plt.tight_layout()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'combined_eval_time.pdf'))
    plt.savefig(os.path.join(output_dir, f'combined_eval_time.pgf'))
    plt.close()

    plt.figure(figsize=(3.4,2.6))
    
    min_t = max(np.min(trad_data['ts']), np.min(nn_data['ts']))
    max_t = min(np.max(trad_data['ts']), np.max(nn_data['ts']))

    t_rel = np.linspace(min_t,max_t,500)

    trad_perf = scipy.interpolate.interp1d(trad_data['ts'], trad_data['eval_time'])
    nn_perf = scipy.interpolate.interp1d(nn_data['ts'], nn_data['eval_time'])

    rel_perf = trad_perf(t_rel) / nn_perf(t_rel)
    
    plt.plot(t_rel, rel_perf, 'k')

    # plt.yscale('log')
    plt.xlabel('Model Time')
    plt.ylabel('Relative Performance')
    plt.title('Relative Performance of Physics-Driven Model', fontsize=9)

    plt.tight_layout()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'relative_perf.pdf'))
    plt.savefig(os.path.join(output_dir, f'relative_perf.pgf'))

    # plt.show()
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
