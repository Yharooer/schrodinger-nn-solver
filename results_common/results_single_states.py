import numpy as np
import os
from utils.numerical_schrodinger import numerical_schrodinger
import torch
import matplotlib.pyplot as plt
from results_common.results_against_time_single_tests import get_tests as get_resuls_against_time_single_tests
from timeit import timeit
import random
import pandas

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'

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
        ts = torch.tensor([t]).float()
        
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


def do_single_states(output_dir, params, model=None):
    tests = get_resuls_against_time_single_tests()

    for test in tests:
        MSE_ERRORS = []
        TIME_TAKEN = []

        xs = np.linspace(0,1,params['GRID_SIZE'])
        ts = list(np.linspace(0,params['MAX_TIME'], params['NUM_TIME_INTERVALS']+1)[1:])
        random.shuffle(ts)

        for t in ts:
            if t == 0:
                MSE_ERRORS.append(0)
                TIME_TAKEN.append(0)
                continue

            print(f'Doing test \'{test.name}\' at time {t}.')
            real_soln, imag_soln = solve_general(params, test, model, xs, t)

            print('TIME FOR SHAPES')
            print(real_soln.shape)

            real_truth = test.real_soln(xs,t)
            imag_truth = test.imag_soln(xs,t)

            mse_loss = np.mean((real_soln - real_truth)**2 + (imag_soln - imag_truth)**2)

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
