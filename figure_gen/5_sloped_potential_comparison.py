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

device = "cuda" if torch.cuda.is_available() else "cpu"

UNSUPERVISED_MODEL = 'google_results/models/103/11/model_at_epoch_1000.pt'
MAX_TIME = 0.48

POTENTIAL_SLOPE = 10

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
    

def test_model(unsup_model, psi0, v, t_max):

    ts = np.linspace(0,MAX_TIME,6)

    # Solve numerically
    print('Solving numerically...')
    num_ys_real,num_ys_imag = solve_single_numerically(psi0, v, ts, 100)
    print('Finished solving numerically.')

    # Solve using our method
    print('Solving unsupervised nn...')
    unsup_nn_ys_real, unsup_nn_ys_imag = solve_single_nn(unsup_model, psi0, v, ts, 100)
    print('Finished unsolving supervised nn.')
    
    xs = np.linspace(0, 1, 100)

    Y_MAX = 1.8

    # Plot animations
    mpl.rcParams['font.family'] = 'EB Garamond'
    mpl.rcParams['font.size'] = 9
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble']="\\usepackage{mathpazo}"

    fig = plt.figure(figsize=(6.8,2.5))

    ## Physics Driven
    
    sp = plt.subplot(2,6,1)
    sp.plot(xs, unsup_nn_ys_real[0,:], color='#0000ff')
    sp.plot(xs, unsup_nn_ys_imag[0,:], color='#ff0000')
    # sp.plot(xs, np.sqrt(unsup_nn_ys_real[0,:]**2 + unsup_nn_ys_imag[0,:]**2), color='#000000')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft='on')
    sp.set_ylabel('Physics-Driven')
    sp.axes.xaxis.set_ticklabels([])
    sp.axes.yaxis.set_ticklabels([])
    sp.title.set_text('$t$={:.2f}'.format(ts[0]))
    sp.set_ylim(-1*Y_MAX, Y_MAX)

    sp = plt.subplot(2,6,2)
    sp.plot(xs, unsup_nn_ys_real[1,:], color='#0000ff')
    sp.plot(xs, unsup_nn_ys_imag[1,:], color='#ff0000')
    # sp.plot(xs, np.sqrt(unsup_nn_ys_real[1,:]**2 + unsup_nn_ys_imag[1,:]**2), color='#000000')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    sp.axes.xaxis.set_ticklabels([])
    sp.axes.yaxis.set_ticklabels([])
    sp.title.set_text('$t$={:.2f}'.format(ts[1]))
    sp.set_ylim(-1*Y_MAX, Y_MAX)


    sp = plt.subplot(2,6,3)
    sp.plot(xs, unsup_nn_ys_real[2,:], color='#0000ff')
    sp.plot(xs, unsup_nn_ys_imag[2,:], color='#ff0000')
    # sp.plot(xs, np.sqrt(unsup_nn_ys_real[2,:]**2 + unsup_nn_ys_imag[2,:]**2), color='#000000')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    sp.axes.xaxis.set_ticklabels([])
    sp.axes.yaxis.set_ticklabels([])
    sp.title.set_text('$t$={:.2f}'.format(ts[2]))
    sp.set_ylim(-1*Y_MAX, Y_MAX)

    sp = plt.subplot(2,6,4)
    sp.plot(xs, unsup_nn_ys_real[3,:], color='#0000ff')
    sp.plot(xs, unsup_nn_ys_imag[3,:], color='#ff0000')
    # sp.plot(xs, np.sqrt(unsup_nn_ys_real[3,:]**2 + unsup_nn_ys_imag[3,:]**2), color='#000000')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    sp.axes.xaxis.set_ticklabels([])
    sp.axes.yaxis.set_ticklabels([])
    sp.title.set_text('$t$={:.2f}'.format(ts[3]))
    sp.set_ylim(-1*Y_MAX, Y_MAX)

    sp = plt.subplot(2,6,5)
    sp.plot(xs, unsup_nn_ys_real[4,:], color='#0000ff')
    sp.plot(xs, unsup_nn_ys_imag[4,:], color='#ff0000')
    # sp.plot(xs, np.sqrt(unsup_nn_ys_real[4,:]**2 + unsup_nn_ys_imag[4,:]**2), color='#000000')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    sp.axes.xaxis.set_ticklabels([])
    sp.axes.yaxis.set_ticklabels([])
    sp.title.set_text('$t$={:.2f}'.format(ts[4]))
    sp.set_ylim(-1*Y_MAX, Y_MAX)

    sp = plt.subplot(2,6,6)
    sp.plot(xs, unsup_nn_ys_real[5,:], color='#0000ff')
    sp.plot(xs, unsup_nn_ys_imag[5,:], color='#ff0000')
    # sp.plot(xs, np.sqrt(unsup_nn_ys_real[5,:]**2 + unsup_nn_ys_imag[5,:]**2), color='#000000')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    sp.axes.xaxis.set_ticklabels([])
    sp.axes.yaxis.set_ticklabels([])
    sp.title.set_text('$t$={:.2f}'.format(ts[5]))
    sp.set_ylim(-1*Y_MAX, Y_MAX)

    # Runge-Kutta

    sp = plt.subplot(2,6,7)
    sp.plot(xs, num_ys_real[0,:], color='#0000ff')
    sp.plot(xs, num_ys_imag[0,:], color='#ff0000')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft='on')
    sp.set_ylabel('Runge–Kutta')
    sp.axes.xaxis.set_ticklabels([])
    sp.axes.yaxis.set_ticklabels([])
    sp.set_ylim(-1*Y_MAX, Y_MAX)

    sp = plt.subplot(2,6,8)
    sp.plot(xs, num_ys_real[1,:], color='#0000ff')
    sp.plot(xs, num_ys_imag[1,:], color='#ff0000')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    sp.axes.xaxis.set_ticklabels([])
    sp.axes.yaxis.set_ticklabels([])
    sp.set_ylim(-1*Y_MAX, Y_MAX)


    sp = plt.subplot(2,6,9)
    sp.plot(xs, num_ys_real[2,:], color='#0000ff')
    sp.plot(xs, num_ys_imag[2,:], color='#ff0000')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    sp.axes.xaxis.set_ticklabels([])
    sp.axes.yaxis.set_ticklabels([])
    sp.set_ylim(-1*Y_MAX, Y_MAX)

    sp = plt.subplot(2,6,10)
    sp.plot(xs, num_ys_real[3,:], color='#0000ff')
    sp.plot(xs, num_ys_imag[3,:], color='#ff0000')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    sp.axes.xaxis.set_ticklabels([])
    sp.axes.yaxis.set_ticklabels([])
    sp.set_ylim(-1*Y_MAX, Y_MAX)

    sp = plt.subplot(2,6,11)
    sp.plot(xs, num_ys_real[4,:], color='#0000ff')
    sp.plot(xs, num_ys_imag[4,:], color='#ff0000')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    sp.axes.xaxis.set_ticklabels([])
    sp.axes.yaxis.set_ticklabels([])
    sp.set_ylim(-1*Y_MAX, Y_MAX)

    sp = plt.subplot(2,6,12)
    sp.plot(xs, num_ys_real[5,:], color='#0000ff')
    sp.plot(xs, num_ys_imag[5,:], color='#ff0000')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    sp.axes.xaxis.set_ticklabels([])
    sp.axes.yaxis.set_ticklabels([])
    sp.set_ylim(-1*Y_MAX, Y_MAX)

    plt.suptitle('Solution to Particle on a Sloped Potential', fontsize=10)

    plt.tight_layout()

    plt.savefig('figure_gen/5_sloped_potential.pdf')
    plt.savefig('figure_gen/5_sloped_potential.pgf')


    plt.show()

    # # Helper for setting up subplot limits and labels
    # def setup_subplot(subplot):
    #     subplot.set_xlim(0,1)
    #     subplot.set_ylim(-2,2)
        
    #     line1, = subplot.plot([],[], lw=2, color='#3333B2')
    #     line2, = subplot.plot([],[], lw=2, color='#B23333')
    #     return line1,line2
    
       
    # subplots = []
    # lines = []
    # for i in range(2):
    #     subplot = plt.subplot(1,2,i+1)
    #     line1, line2 = setup_subplot(subplot)
    #     lines.append(line1)
    #     lines.append(line2)
    #     subplots.append(subplot)
        
    # subplots[0].title.set_text('Physics-Driven Model')
    # subplots[0].set_ylabel("$\\psi(\\vec x,t)$")
    # subplots[1].title.set_text('Numerical Model')
    
    # def animate(i):
    #     lines[0].set_data(xs, nn_ys_real[i,:])
    #     lines[1].set_data(xs, nn_ys_imag[i,:])
        
    #     lines[2].set_data(xs, num_ys_real[i,:])
    #     lines[3].set_data(xs, num_ys_imag[i,:])
    
    # for i in range(len(ts)):
    #     animate(i)
    #     plt.savefig(output_file + '-' + str(i) + '.pdf')
    
    # plt.close()


class ModelTests():
    def __init__(self, t_max):
        self.tests = []
        self.names = set()
        self.t_max = t_max

    def add(self, name, psi0_real, psi0_imag, v0):
        if name in self.names:
            raise ValueError('Name of test already used.')
        self.tests.append((name, psi0_real, psi0_imag, v0))

    def perform_tests(self, model):
        for test in self.tests:
            test_model(model, lambda x: (test[1](x), test[2](x)), test[3], self.t_max)


def main():
    params = {
        'MODEL': UNSUPERVISED_MODEL,
        'T_MAX': 0.5
    }

    for font in mpl.font_manager.findSystemFonts('fonts'):
        mpl.font_manager.fontManager.addfont(font)

    # Load model
    checkpoint = torch.load(params['MODEL'], map_location=torch.device('cpu'))
    model_state_dict = checkpoint['model_state_dict']
    hidden_dim = checkpoint['params']['HIDDEN_LAYER_SIZE']
    num_layers = checkpoint['params']['NUM_HIDDEN_LAYERS'] if 'params' in checkpoint and 'NUM_HIDDEN_LAYERS' in checkpoint['params'] else 2

    model = SchrodingerModel(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    model.load_state_dict(model_state_dict)

    print(f"Loaded model with {num_layers} hidden layers with {hidden_dim} nodes each.")

    # Test model
    test = ModelTests(params['T_MAX'])
    # test.add('sloped_box_0', lambda x: np.sqrt(2)*(0.480414*np.sin(np.pi*x) + 0.707106*np.sin(2*np.pi*x) + 0.518847*np.sin(3*np.pi*x)), lambda x: 0*x, lambda x: 8*x)
    test.add('sloped_box_1', lambda x: np.sqrt(2)*np.sin(np.pi*x), lambda x: 0*x, lambda x: POTENTIAL_SLOPE*x - POTENTIAL_SLOPE/2 - np.pi**2/2)

    test.perform_tests(model)
    

if __name__ == "__main__":
    main()
