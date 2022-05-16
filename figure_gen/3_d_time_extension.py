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
import matplotlib as mpl
import json
import pandas
from tqdm import tqdm

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'

NUM_NUMERICAL = 5

def solve_numerically(psi0, test):
    t = test[0]

    initials = np.zeros((3, 100, 1))
    initials[0, :, 0], initials[1, :, 0] = psi0(np.linspace(0,1,100))
    initials[2, :, 0] = 0

    soln = numerical_schrodinger(initials, [t], grid_size=100)
    real_soln = soln[0,:,0,0]
    imag_soln = soln[1,:,0,0]

    return real_soln, imag_soln

def do_generalisation_tests_and_plots(max_test_fourier_mode, num_tests, supervised_models, unsupervised_models, plot_title, plot_xlabel, figure_save_location, is_log=False):

    def generate_tests(num, max_test_time, min_test_time):
        # Generate tests
        tests = [] # Size five tuple of (t), (real cffs), (imaginary cffs), (sampled real analyticial solution), (sampled imag analyticial solution)
        for i in range(num):
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
        return tests


    # Now evaluate the different models
    DATA_DRIVEN_X = []
    DATA_DRIVEN_MEANS = []
    DATA_DRIVEN_DELTAS = []

    PHYSICS_DRIVEN_X = []
    PHYSICS_DRIVEN_MEANS = []
    PHYSICS_DRIVEN_DELTAS = []


    def solve_single_nn(model, psi0, v, ts, grid_size=100):
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


    def eval_model(model_path, max_test_time):
        print(f'Evaluating model \'{model_path}\'')

        # Load model
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model_state_dict = checkpoint['model_state_dict']
        hidden_dim = checkpoint['params']['HIDDEN_LAYER_SIZE']
        num_layers = checkpoint['params']['NUM_HIDDEN_LAYERS'] if 'params' in checkpoint and 'NUM_HIDDEN_LAYERS' in checkpoint['params'] else 2

        model = SchrodingerModel(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
        model.load_state_dict(model_state_dict)

        # Evalulate results
        mse_errors = []

        for test in generate_tests(num_tests,max_test_time, 0):
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


    def eval_numerical(num_tests,max_test_time):
        mse_errors = []
        print(f'Starting numerical integration benchmarks for time {max_test_time}')
        for test in tqdm(generate_tests(num_tests,max_test_time, 0)):
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

        return np.mean(mse_errors), (np.std(mse_errors))/np.sqrt(len(mse_errors))


    for d_model in supervised_models:
        avg, delta = eval_model(d_model[1], d_model[0])
        DATA_DRIVEN_X.append(d_model[0])
        DATA_DRIVEN_MEANS.append(avg)
        DATA_DRIVEN_DELTAS.append(delta)

    DATA_DRIVEN_MEANS = np.array(DATA_DRIVEN_MEANS)
    DATA_DRIVEN_DELTAS = np.array(DATA_DRIVEN_DELTAS)

    for p_model in unsupervised_models:
        avg, delta = eval_model(p_model[1], p_model[0])
        PHYSICS_DRIVEN_X.append(p_model[0])
        PHYSICS_DRIVEN_MEANS.append(avg)
        PHYSICS_DRIVEN_DELTAS.append(delta)

    NUMERICAL_TIMES = np.linspace(0.1,1.0,5)
    NUMERICAL_MEANS = []
    NUMERICAL_DELTAS = []
    for i in NUMERICAL_TIMES:
        print(f'Doing numerical time {i}')
        avg, delta = eval_numerical(NUM_NUMERICAL, i)
        NUMERICAL_MEANS.append(avg)
        NUMERICAL_DELTAS.append(delta)


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
    nline = plt.errorbar(NUMERICAL_TIMES, NUMERICAL_MEANS, yerr=NUMERICAL_DELTAS, color='#8000ff')
    dline = plt.errorbar(DATA_DRIVEN_X, DATA_DRIVEN_MEANS, yerr=DATA_DRIVEN_DELTAS, color='#ff8000')
    pline = plt.errorbar(PHYSICS_DRIVEN_X, PHYSICS_DRIVEN_MEANS, yerr=PHYSICS_DRIVEN_DELTAS, color='#00b35a')
    plt.xlabel(plot_xlabel)
    plt.ylabel('MSE Error')
    if is_log:
        plt.yscale('log')
    plt.title(plot_title, fontsize=9, y=1.25)

    leg = plt.legend([dline, pline, nline], ['Data-Drvn', 'Physics-Drvn', 'Numerical'], shadow=False, edgecolor='#000000', framealpha=1.0, ncol=3, bbox_to_anchor=(0,1.02,1,0.2), loc='lower left', mode='expand', prop={'size': 7})
    leg.get_frame().set_boxstyle('Square', pad=0.1)
    leg.get_frame().set(capstyle='butt', joinstyle='miter', linewidth=1.0)

    plt.tight_layout()
    plt.tight_layout()

    plt.savefig(f'{figure_save_location}.pdf')
    plt.savefig(f'{figure_save_location}.pgf')




TESTING_MAX_FOURIER_MODE = 2
NUM_TESTS = 100

# TODO we need to convert these to RMS potentials
DATA_DRIVEN_FOURIER_MODE_MODELS = [
    (0.1, 'kiiara_2_results/models/37/1/model_at_epoch_1000.pt'),
    (0.3, 'kiiara_2_results/models/38/1/model_at_epoch_860.pt'),
    (0.5, 'kiiara_2_results/models/39/1/model_at_epoch_1000.pt'),
    (0.7, 'kiiara_2_results/models/40/1/model_at_epoch_1000.pt'),
    (0.9, 'kiiara_2_results/models/41/1/model_at_epoch_1000.pt'),
]

PHYSICS_DRIVEN_FOURIER_MODE_MODELS = [
    (0.1, 'google_results/models/90/13/model_at_epoch_650.pt'),
    (0.3, 'google_results/models/90/14/model_at_epoch_1000.pt'),
    (0.5, 'google_results/models/90/15/model_at_epoch_1000.pt'),
    (0.7, 'google_results/models/90/29/model_at_epoch_1000.pt'),
    (0.9, 'google_results/models/90/30/model_at_epoch_1000.pt'),
    # (1.5, 'google_results/models/90/31/model_at_epoch_870.pt'),
    # (2.5, 'google_results/models/90/32/model_at_epoch_1000.pt'),
]

do_generalisation_tests_and_plots(
    max_test_fourier_mode=TESTING_MAX_FOURIER_MODE,
    num_tests=NUM_TESTS,
    supervised_models=DATA_DRIVEN_FOURIER_MODE_MODELS,
    unsupervised_models=PHYSICS_DRIVEN_FOURIER_MODE_MODELS,
    plot_title='Generalisation To Larger Times',
    plot_xlabel='Time Domain Length',
    figure_save_location='figure_gen/3_d_time_extension'
)
