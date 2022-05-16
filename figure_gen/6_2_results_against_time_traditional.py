import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
import os
import argparse
from utils.argparser_helper import reduce_to_single_arguments, check_args_positive_numbers, print_arguments
import matplotlib.pyplot as plt
from results_common.results_against_time_single_tests import ResultsAgainstTimeSingleTest
from utils.numerical_schrodinger import numerical_schrodinger
import json
from tqdm import tqdm
from timeit import timeit
import pandas

device = 'cpu'

FILENAME='figure_gen/sloped_better_soln.npy'

POTENTIAL_SLOPE = 10

def solve_numerically(params, test, xs, t):
    initials = np.zeros((3, params['GRID_SIZE'], 1))
    initials[0, :, 0] = test.psi0_real(xs)
    initials[1, :, 0] = test.psi0_imag(xs)
    initials[2, :, 0] = test.v(xs)

    soln = numerical_schrodinger(initials, [t], grid_size=params['GRID_SIZE'])
    real_soln = soln[0,:,0,0]
    imag_soln = soln[1,:,0,0]

    return real_soln, imag_soln

def solve_nn(params, test, model, xs, t):

    t_steps = np.arange(0,t,params['MODEL_MAX_TIME'])[1:]
    if len(t_steps) == 0 or t_steps[-1] != t:
        t_steps = np.append(t_steps, t)

    real_current = torch.tensor(test.psi0_real(xs))
    imag_current = torch.tensor(test.psi0_imag(xs))

    vs = torch.tensor(test.v(xs))

    t_so_far = 0
    for t_i in t_steps:
        t_next = t_i - t_so_far
        t_so_far = t_i

        xs = torch.linspace(0, 1, 100).float()
        ts = torch.tensor([t_next]).float()
        
        if not torch.is_tensor(ts):
            ts = torch.tensor(ts)
            
        xts = torch.cartesian_prod(xs,ts)
        
        nn_in = torch.zeros((len(xts), 300 + 2))
        nn_in[:,0:2] = xts
        nn_in[:,2:] = torch.cat((real_current, imag_current, vs))
        nn_in = nn_in.to(device)
        
        nn_out = model(nn_in).detach()
        
        real_current = torch.reshape(nn_out[:,0], (100, ))
        imag_current = torch.reshape(nn_out[:,1], (100, ))

    return real_current.numpy(), imag_current.numpy()


def solve_general(params, test, model, xs, t):
    if params['SIMULATION_TYPE'] == 'NUMERICAL':
        return solve_numerically(params, test, xs, t)
    elif params['SIMULATION_TYPE'] == 'NN':
        return solve_nn(params, test, model, xs, t)
    else:
        raise ValueError('\'SIMULATION_TYPE\' must be \'NUMERICAL\' or \'NN\'.')


def do_single_states(test, output_dir, params, model=None):

    load_data = np.load(FILENAME, allow_pickle=True)
    xs = load_data.item().get('xs')
    ts = load_data.item().get('ts')
    soln = load_data.item().get('soln')

    MSE_ERRORS = []
    TIME_TAKEN = []

    ts_orig = np.copy(ts)

    np.random.shuffle(ts)

    for t in tqdm(ts):
        if t == 0:
            MSE_ERRORS.append(0)
            TIME_TAKEN.append(0)
            continue

        print(f'Doing test \'{test.name}\' at time {t}.')
        real_soln, imag_soln = solve_general(params, test, model, xs, t)

        real_truth = soln[0,:,0,np.where(ts_orig == t)[0]]
        imag_truth = soln[1,:,0,np.where(ts_orig == t)[0]]

        mse_loss = np.mean((real_soln - real_truth[0,:])**2 + (imag_soln - imag_truth[0,:])**2)

        print(mse_loss)

        # plt.plot(xs,real_soln, color='#ff0000')
        # plt.plot(xs, real_truth[0,:], color='#aa0000')
        # plt.plot(xs, imag_soln, color='#00ff00')
        # plt.plot(xs, imag_truth[0,:], color='#00aa00')
        
        # plt.show()

        time = timeit(lambda: solve_general(params, test, model, xs, t), number=params['ITERATIONS'])/params['ITERATIONS']

        MSE_ERRORS.append(mse_loss)
        TIME_TAKEN.append(time)

    idx = np.argsort(ts)
    ts = np.array(ts)[idx]
    MSE_ERRORS = np.array(MSE_ERRORS)[idx]
    TIME_TAKEN = np.array(TIME_TAKEN)[idx]

    df = pandas.DataFrame({'ts': ts, 'mse_error': MSE_ERRORS, 'eval_time': TIME_TAKEN})

    model_type_name = 'Numerical Model' if params['SIMULATION_TYPE'] == 'NUMERICAL' else 'Our Model' if params['SIMULATION_TYPE'] == 'NN' else 'Unknown Model'

    plt.figure(figsize=(6,6))
    plt.plot(ts, MSE_ERRORS)
    plt.xlabel('Simulation Time')
    plt.ylabel('MSE Error')
    plt.title(f'Error Accumulation for {model_type_name} for \'{test.name}\'')
    plt.savefig(os.path.join(output_dir, f'{test.name}_errors.pdf'))
    plt.close()

    plt.figure(figsize=(6,6))
    plt.plot(ts, TIME_TAKEN)
    plt.xlabel('Simulation Time')
    plt.ylabel('Evaluation Time')
    plt.title(f'Evaluation Time for {model_type_name} for \'{test.name}\'')
    plt.savefig(os.path.join(output_dir, f'{test.name}_eval_time.pdf'))
    plt.close()

    df.to_csv(os.path.join(output_dir, f'{test.name}_data.csv'))

def get_arguments():
    parser = argparse.ArgumentParser(description="Tests performance and speed of numerical solution.")
    parser.add_argument('--GRID_SIZE', type=int, nargs='?', default=100,
                        help='Grid size of the model to evaluate.')
    parser.add_argument('--ITERATIONS', type=int, nargs='?', default=5,
                        help='(Optional) The number of iterations to take for each timestep. Defaults to 5.')                  

    args = vars(parser.parse_args())
    reduce_to_single_arguments(args)
    check_arguments(args)
    print_arguments(args)
    return args


def check_arguments(args):
    check_args_positive_numbers(args, ['GRID_SIZE'])


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

    test = ResultsAgainstTimeSingleTest('slope', lambda x: np.sqrt(2)*np.sin(np.pi*x), lambda x: 0*x, lambda x: POTENTIAL_SLOPE*x - POTENTIAL_SLOPE/2 - np.pi**2/2, None, None)

    do_single_states(test, output_directory, params)
    

if __name__ == "__main__":
    main()
