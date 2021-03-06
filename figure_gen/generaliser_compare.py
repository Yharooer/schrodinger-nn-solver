import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
import argparse
from utils.argparser_helper import reduce_to_single_arguments, check_args_positive_numbers, print_arguments
from utils.numerical_schrodinger import numerical_schrodinger
from utils.model_definition import SchrodingerModel
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib as mpl
import json

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'

NUMERICAL_SUBSET = 15

def do_generalisation_tests_and_plots(max_test_fourier_mode, min_test_time, max_test_time, num_tests, supervised_models, unsupervised_models, plot_title, plot_xlabel, figure_save_location, is_log=False, include_zero_benchmark=True):

    # Generate tests
    tests = [] # Size five tuple of (t), (real cffs), (imaginary cffs), (sampled real analyticial solution), (sampled imag analyticial solution)
    for i in range(num_tests):
        t = (np.random.rand(1)*(max_test_time-min_test_time) + min_test_time)[0]

        real_cffs = 2*np.random.rand(max_test_fourier_mode) - 1 
        imag_cffs = 2*np.random.rand(max_test_fourier_mode) - 1

        scale_factor = np.sum(real_cffs**2) + np.sum(imag_cffs**2)
        scale_factor = np.sqrt(2/scale_factor)
        real_cffs *= scale_factor
        imag_cffs *= scale_factor

        # print(np.sum(real_cffs**2 + imag_cffs**2))

        basis_n = lambda n,x: np.sin(n*np.pi*x)
        freq_n = lambda n: 0.5 * n**2 * np.pi**2

        amplitudes = np.sqrt(real_cffs**2 + imag_cffs**2)
        phases = np.arctan2(imag_cffs, real_cffs)

        final_phases = phases - freq_n(np.arange(1,max_test_fourier_mode+1))*t

        xs = np.linspace(0,1,100)
        final_real = np.zeros(100)
        final_imag = np.zeros(100)

        for i in range(max_test_fourier_mode):
            n_i  = i+1
            final_real += basis_n(n_i, xs)*np.cos(final_phases[i])
            final_imag += basis_n(n_i, xs)*np.sin(final_phases[i])

        tests.append((t, real_cffs, imag_cffs, final_real, final_imag))

    # Evaluate benchmark
    def solve_numerically(psi0, test):
        t = test[0]

        initials = np.zeros((3, 100, 1))
        initials[0, :, 0], initials[1, :, 0] = psi0(np.linspace(0,1,100))
        initials[2, :, 0] = 0

        soln = numerical_schrodinger(initials, [t], grid_size=100)
        real_soln = soln[0,:,0,0]
        imag_soln = soln[1,:,0,0]

        return real_soln, imag_soln

    mse_errors = []
    if NUMERICAL_SUBSET > 0:
        print('Starting numerical integration benchmarks')
        for test in tqdm(tests[0:NUMERICAL_SUBSET]):
            def psi0(xs):
                basis_n = lambda n,x: np.sin(n*np.pi*x)
                real_psi0 = xs*0
                imag_psi0 = xs*0
                for i in range(max_test_fourier_mode):
                    n_i = i+1
                    real_psi0 += test[1][i]*basis_n(n_i, xs)
                    imag_psi0 += test[2][i]*basis_n(n_i, xs)
                return real_psi0, imag_psi0

            out_real, out_imag = solve_numerically(psi0, test)

            mse_loss = np.sum((out_real - test[3])**2 + (out_imag - test[4])**2)/200

            mse_errors.append(mse_loss)
        print('Finishing numerical integration benchmark')

        benchmark = np.mean(mse_errors)
    else:
        benchmark = 1 # For testing


    # Now evaluate the different models
    DATA_DRIVEN_X = []
    DATA_DRIVEN_MEANS = []
    DATA_DRIVEN_DELTAS = []

    PHYSICS_DRIVEN_X = []
    PHYSICS_DRIVEN_MEANS = []
    PHYSICS_DRIVEN_DELTAS = []


    def solve_single_nn(model, psi0, v, ts, grid_size=100):
        if model == None:
            return np.zeros((1,100)), np.zeros((1,100))

        xs = torch.linspace(0, 1, grid_size).float()
        ts = torch.tensor(ts).float()
        p0_real, p0_imag = psi0(xs)
        vs = v(xs)

        if not torch.is_tensor(ts):
            ts = torch.tensor(ts)
            
        xts = torch.cartesian_prod(xs,ts)
        
        nn_in = torch.zeros((len(xts), 3*grid_size + 2))
        nn_in[:,0:2] = xts
        nn_in[:,2:] = torch.cat((p0_real, p0_imag, vs))
        nn_in = nn_in.to(device)
        
        nn_out = model(nn_in).cpu().detach().numpy()
        
        out_real = np.reshape(nn_out[:,0], (grid_size, len(ts))).T
        out_imag = np.reshape(nn_out[:,1], (grid_size, len(ts))).T 
        
        return out_real, out_imag


    def eval_model(model_path):
        print(f'Evaluating model \'{model_path}\'')

        # Load model
        if model_path != None:
            checkpoint = torch.load(model_path, map_location=torch.device(device))
            model_state_dict = checkpoint['model_state_dict']
            hidden_dim = checkpoint['params']['HIDDEN_LAYER_SIZE']
            num_layers = checkpoint['params']['NUM_HIDDEN_LAYERS'] if 'params' in checkpoint and 'NUM_HIDDEN_LAYERS' in checkpoint['params'] else 2

            model = SchrodingerModel(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
            model.load_state_dict(model_state_dict)
        else:
            model = None

        # Evalulate results
        mse_errors = []

        for test in tests:
            def psi0(xs):
                basis_n = lambda n,x: np.sin(n*np.pi*x)
                real_psi0 = xs*0
                imag_psi0 = xs*0
                for i in range(max_test_fourier_mode):
                    n_i = i+1
                    real_psi0 += test[1][i]*basis_n(n_i, xs)
                    imag_psi0 += test[2][i]*basis_n(n_i, xs)
                return real_psi0, imag_psi0

            out_real, out_imag = solve_single_nn(model, psi0, lambda x: 0*x, [test[0]])
            out_real = out_real[0,:]
            out_imag = out_imag[0,:]

            mse_loss = np.sum((out_real - test[3])**2 + (out_imag - test[4])**2)/200

            mse_errors.append(mse_loss)

        return np.mean(mse_errors), (np.std(mse_errors))/np.sqrt(len(mse_errors))


    for d_model in supervised_models:
        avg, delta = eval_model(d_model[1])
        DATA_DRIVEN_X.append(d_model[0])
        DATA_DRIVEN_MEANS.append(avg)
        DATA_DRIVEN_DELTAS.append(delta)

    DATA_DRIVEN_MEANS = np.array(DATA_DRIVEN_MEANS)
    DATA_DRIVEN_DELTAS = np.array(DATA_DRIVEN_DELTAS)

    for p_model in unsupervised_models:
        avg, delta = eval_model(p_model[1])
        PHYSICS_DRIVEN_X.append(p_model[0])
        PHYSICS_DRIVEN_MEANS.append(avg)
        PHYSICS_DRIVEN_DELTAS.append(delta)

    # Zero model
    zero_avg, zero_delta = eval_model(None)
    print(f'Zero is {zero_avg}.')

    PHYSICS_DRIVEN_MEANS = np.array(PHYSICS_DRIVEN_MEANS)
    PHYSICS_DRIVEN_DELTAS = np.array(PHYSICS_DRIVEN_DELTAS)


    # Plot results!
    for font in mpl.font_manager.findSystemFonts('fonts'):
        mpl.font_manager.fontManager.addfont(font)

    mpl.rcParams['font.family'] = 'EB Garamond'
    mpl.rcParams['font.size'] = 9
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble']="\\usepackage{mathpazo}"

    fig = plt.figure(figsize=(3.4,2.6))
    dplt = plt.errorbar(DATA_DRIVEN_X, DATA_DRIVEN_MEANS, yerr=DATA_DRIVEN_DELTAS, color='#ff8000')
    pplt = plt.errorbar(PHYSICS_DRIVEN_X, PHYSICS_DRIVEN_MEANS, yerr=PHYSICS_DRIVEN_DELTAS, color='#00b35a')
    # if NUMERICAL_SUBSET > 0:
    bplt = plt.axhline(y=benchmark, color='#8000ff', linestyle='--')
    plt.xlabel(plot_xlabel)
    plt.ylabel('MSE Error')
    if is_log:
        plt.yscale('log')
    plt.title(plot_title, fontsize=9, y=1.25)

    handles = [dplt, pplt]
    if NUMERICAL_SUBSET > -5:
        handles.append(bplt)
    
    leg = plt.legend(handles, ['Data-Drvn', 'Physics-Drvn', 'Numerical'], shadow=False, edgecolor='#000000', framealpha=1.0, ncol=3, bbox_to_anchor=(0,1.0,1,0.0), loc='lower left', mode='expand', prop={'size': 7})
    leg.get_frame().set_boxstyle('Square', pad=0.1)
    leg.get_frame().set(capstyle='butt', joinstyle='miter', linewidth=1.0)

    plt.tight_layout()
    plt.tight_layout()

    plt.savefig(f'{figure_save_location}.pdf')
    plt.savefig(f'{figure_save_location}.pgf')
