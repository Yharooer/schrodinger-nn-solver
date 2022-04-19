import numpy as np
import os
import argparse
from utils.argparser_helper import reduce_to_single_arguments, check_args_positive_numbers, print_arguments
from utils.numerical_schrodinger import numerical_schrodinger
from utils.model_definition import SchrodingerModel
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

def solve_single_numerically(psi0, v, ts, grid_size=100):
    xs_num = np.linspace(0,1,grid_size)
    p0_real, p0_imag = psi0(xs_num)
    vs = v(xs_num)
    
    initials = np.zeros((3, grid_size, 1))
    initials[0, :, 0] = p0_real.T
    initials[1, :, 0] = p0_imag.T
    initials[2, :, 0] = vs.T
    
    num_y = numerical_schrodinger(initials, ts, grid_size=100)
    
    num_ys_real = num_y[0,:,0,:]
    num_ys_imag = num_y[1,:,0,:]
    
    return num_ys_real.T, num_ys_imag.T


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
    

def test_model(model, psi0, v, t_max, t_steps, output_file, plot_phase=False):
    ts = np.linspace(0, t_max, t_steps)

    # Solve numerically
    print('Solving numerically...')
    num_ys_real,num_ys_imag = solve_single_numerically(psi0, v, ts, 100)
    print('Finished solving numerically.')
   
    # Solve using our method
    print('Solving nn...')
    nn_ys_real, nn_ys_imag = solve_single_nn(model, psi0, v, ts, 100)
    print('Finished solving nn.')
    
    xs = np.linspace(0, 1, 100)
    
    # Normalisation vs time
    fig = plt.figure(figsize=(5,5))
    plt.plot(ts, np.sum(nn_ys_real**2 + nn_ys_imag**2, axis=1) / 100)
    fig.suptitle('Normalisation')
    plt.xlabel('Time')
    plt.ylabel('∫|Ψ|² dx')
    
    # Plot animations
    fig = plt.figure(figsize=(12,8))
    
    plt.rcParams["animation.html"] = "html5"
    plt.rcParams["figure.dpi"] = 75
    
    # Helper for setting up subplot limits and labels
    def setup_subplot(subplot, prop):
        prop = prop.upper()
        if not prop in ["REAL", "IMAG", "ABS", "PHASE"]:
            raise ValueError(f'Bad property \'{prop}\'.')
        subplot.set_xlim(0,1)
        if prop in ["REAL", "IMAG"]:
            subplot.set_ylim(-2,2)
            subplot.set_ylabel("Real" if prop == "REAL" else "Imaginary")
        if prop == "ABS":
            subplot.set_ylim(0,2)
            subplot.set_ylabel("Magnitude")
        if prop == "PHASE":
            subplot.set_ylim(-np.pi,np.pi)
            subplot.set_ylabel("Phase")
            subplot.set_yticks(np.arange(-np.pi,np.pi,np.pi/4))
        
        line, = subplot.plot([],[], lw=2)
        return line
    
    # Types of each subplot
    props = [None]*4
    if plot_phase:
        props = ["ABS", "ABS", "PHASE", "PHASE"]
    else:
        props = ["REAL", "REAL", "IMAG", "IMAG"]
       
    subplots = [None]*4
    lines = [None]*4
    for i in range(4):
        subplots[i] = plt.subplot(2,2,i+1)
        lines[i] = setup_subplot(subplots[i], props[i])
        
    subplots[0].title.set_text('NN model')
    subplots[1].title.set_text('Numerical model')
    
    def animate(i):
        if plot_phase:
            lines[0].set_data(xs, np.sqrt(nn_ys_real[i,:]**2 + nn_ys_imag[i,:]**2))
            lines[2].set_data(xs, np.arctan2(nn_ys_real[i,:], nn_ys_imag[i,:]))
        
            lines[1].set_data(xs, np.sqrt(num_ys_real[i,:]**2 + num_ys_imag[i,:]**2))
            lines[3].set_data(xs, np.arctan2(num_ys_real[i,:], num_ys_imag[i,:]))

        else:
            lines[0].set_data(xs, nn_ys_real[i,:])
            lines[2].set_data(xs, nn_ys_imag[i,:])
            
            lines[1].set_data(xs, num_ys_real[i,:])
            lines[3].set_data(xs, num_ys_imag[i,:])

        return lines
    
    anim = animation.FuncAnimation(fig, animate, frames=len(ts), interval=50, blit=True)
    # writer = animation.Writers['ffmpeg'](fps=30, bitrate=1800)
    anim.save(output_file)


def get_arguments():
    parser = argparse.ArgumentParser(description="Tests model against samples.")
    parser.add_argument('MODEL', type=str, nargs=1,
                        help='The model to evaluate.')
    parser.add_argument('T_MAX', type=float, nargs=1,
                        help='The max time step.')
    parser.add_argument('--T_STEPS', type=int, nargs='?', default=200,
                        help='The max number of steps.')
    parser.add_argument('--HIDDEN_LAYER_SIZE', type=int, nargs='?', default=500,
                        help='The number of hidden layers.')

    args = vars(parser.parse_args())
    reduce_to_single_arguments(args)
    check_arguments(args)
    print_arguments(args)
    return args


def check_arguments(args):
    if not os.path.isfile(args['MODEL']):
        raise ValueError(f'Cannot find model file \'{args["MODEL"]}\'.')

    check_args_positive_numbers(args, ['T_MAX', 'T_STEPS', 'HIDDEN_LAYER_SIZE'])


def get_output_folder():
    if not os.path.isdir('model_evals'):
        print('Making directory \'model_evals\'.')
        os.mkdir('model_evals')

    num = 1
    while True:
        directory = f'model_evals/{num}'
        if not os.path.exists(directory):
            print(f'Making directory \'{directory}\'.')
            os.mkdir(directory)
            return directory
        num += 1


def create_params_file(args, directory):
    with open(f'{directory}/params.json', 'w') as params_file:
        params_file.write(json.dumps(args))


class ModelTests():
    def __init__(self, t_max, t_steps, output_dir):
        self.tests = []
        self.names = set()
        self.t_max = t_max
        self.t_steps = t_steps
        self.output_dir = output_dir

    def add(self, name, psi0_real, psi0_imag, v0):
        if name in self.names:
            raise ValueError('Name of test already used.')
        self.tests.append((name, psi0_real, psi0_imag, v0))

    def perform_tests(self, model):
        for test in self.tests:
            test_model(model, lambda x: (test[1](x), test[2](x)), test[3], self.t_max, self.t_steps, os.path.join(self.output_dir, test[0]+'.mp4'))


def main():
    params = get_arguments()

    # Load model
    model_state_dict = torch.load(params['MODEL'])['model_state_dict']
    # print(model_state_dict)
    model = SchrodingerModel(hidden_dim=params['HIDDEN_LAYER_SIZE']).to(device)
    model.load_state_dict(model_state_dict)

    # Get output directory
    output_directory = get_output_folder()
    create_params_file(params, output_directory)

    # Test model
    test = ModelTests(params['T_MAX'], params['T_STEPS'], output_directory)
    test.add('particle_in_box_eigen1', lambda x: np.sqrt(2)*np.sin(np.pi*x), lambda x: 0*x, lambda x: 0*x)
    test.add('particle_in_box_eigen2', lambda x: np.sqrt(2)*np.sin(2*np.pi*x), lambda x: 0*x, lambda x: 0*x)
    test.add('particle_in_box_eigen3', lambda x: np.sqrt(2)*np.sin(3*np.pi*x), lambda x: 0*x, lambda x: 0*x)
    test.add('particle_in_box_eigen4', lambda x: np.sqrt(2)*np.sin(4*np.pi*x), lambda x: 0*x, lambda x: 0*x)
    test.add('raised_box_eigen1', lambda x: np.sqrt(2)*np.sin(np.pi*x), lambda x: 0*x, lambda x: 0*x + 5)
    test.add('raised_box_eigen2', lambda x: np.sqrt(2)*np.sin(2*np.pi*x), lambda x: 0*x, lambda x: 0*x + 5)
    test.add('raised_box_eigen3', lambda x: np.sqrt(2)*np.sin(3*np.pi*x), lambda x: 0*x, lambda x: 0*x + 5)
    test.add('sloped_box_1', lambda x: np.sqrt(2)*np.sin(np.pi*x), lambda x: 0*x, lambda x: 0.2*x)
    test.add('sloped_box_2', lambda x: np.sqrt(2)*np.sin(np.pi*x), lambda x: 0*x, lambda x: x)
    test.add('harmonic_1', lambda x: 1.728*np.exp(-0.5*7*(2*x-1)**2), lambda x: 0*x, lambda x: 0.125*7*(2*x-1)**2)
    # test.add('particle_in_box_flat2', lambda x: (1-(1/3))*(1-(2*x-1)**2), lambda x: 0*x, lambda x: 0*x)
    # test.add('particle_in_box_flat4', lambda x: (1-(1/5))*(1-(2*x-1)**4), lambda x: 0*x, lambda x: 0*x)
    # test.add('particle_in_box_flat8', lambda x: (1-(1/9))*(1-(2*x-1)**8), lambda x: 0*x, lambda x: 0*x)
    # test.add('particle_in_box_flat16', lambda x: (1-(1/17))*(1-(2*x-1)**16), lambda x: 0*x, lambda x: 0*x)

    test.perform_tests(model)
    

if __name__ == "__main__":
    main()
