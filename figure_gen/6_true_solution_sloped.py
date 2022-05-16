import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

device = 'cpu'

import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from results_common.results_against_time_single_tests import ResultsAgainstTimeSingleTest
from timeit import timeit
import random
import pandas
import scipy.interpolate
import time
from matplotlib import animation
import scipy.integrate
from tqdm import tqdm

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'

POTENTIAL_SLOPE = 10
GRID_SIZE = 300
TOLERANCES = (1e-3, 1e-6)


def numerical_schrodinger(initials, ts, grid_size=100, grid_length=1):

    psi0 = initials[0:2, :, :]    
    v = initials[2, :, :]
    shape = psi0.shape
    flattened_shape = np.prod(shape)
    
    # flatten
    psi0 = np.reshape(psi0, flattened_shape)
    
    # construct laplacian operator and then Hamiltonian
    dx = grid_length/grid_size
    D2 = -2*np.eye(grid_size)
    for i in range(grid_size-1):
        D2[i,i+1] = 1 
        D2[i+1,i] = 1
    
    KE = -0.5*D2/(dx**2)
 
    def dpsi_dt(t,y):        
        y = np.reshape(y, shape)
        psi_real = y[0]
        psi_imag = y[1]
        dpsi_real = np.expand_dims(KE@psi_imag + v*psi_imag, 0)
        dpsi_imag = np.expand_dims(-KE@psi_real - v*psi_real, 0)
        return np.reshape(np.concatenate((dpsi_real, dpsi_imag), axis=0), flattened_shape)
    
    sol = scipy.integrate.solve_ivp(dpsi_dt, t_span=[0,np.max(ts)], y0=psi0, t_eval=ts, method="RK23", rtol=TOLERANCES[0], atol=TOLERANCES[1])
    
    return np.reshape(sol.y, shape+(len(ts),))

def solve_numerically(grid_size, test, xs, ts):
    initials = np.zeros((3, grid_size, 1))
    initials[0, :, 0] = test.psi0_real(xs)
    initials[1, :, 0] = test.psi0_imag(xs)
    initials[2, :, 0] = test.v(xs)

    soln = numerical_schrodinger(initials, ts, grid_size=grid_size)
    
    return soln

start = time.perf_counter()

test = ResultsAgainstTimeSingleTest('slope', lambda x: np.sqrt(2)*np.sin(np.pi*x), lambda x: 0*x, lambda x: POTENTIAL_SLOPE*x - POTENTIAL_SLOPE/2 - np.pi**2/2, None, None)
xs = np.linspace(0,1,GRID_SIZE)
ts = np.linspace(0,1.5,300)
print('Starting Runge-Kutta integration')
solution = solve_numerically(GRID_SIZE, test, xs, ts)
print('Ending Runge-Kutta integration')

end = time.perf_counter()

xs_train_grid = np.linspace(0,1,100)
solution = scipy.interpolate.interp1d(np.linspace(0,1,GRID_SIZE), solution, kind='linear', axis=1)(xs_train_grid)

print(f'Took {end-start} seconds.')

print('Saving')
np.save('figure_gen/sloped_better_soln.npy', {
    'xs': xs_train_grid,
    'ts': ts,
    'soln': solution
})
print('Finished saving data')

print('Generating little movie')

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
props = ["REAL", "REAL", "IMAG", "IMAG"]
    
subplots = [None]*4
lines = [None]*4
for i in range(4):
    subplots[i] = plt.subplot(2,2,i+1)
    lines[i] = setup_subplot(subplots[i], props[i])
    
subplots[0].title.set_text('NN model')
subplots[1].title.set_text('Numerical model')

def animate(i):
    lines[0].set_data(xs_train_grid, solution[0,:,0,i])
    lines[2].set_data(xs_train_grid, solution[1,:,0,i])
    subplots[0].axes.set_title(f't={ts[i]}')

    return lines

anim = animation.FuncAnimation(fig, animate, frames=len(ts), interval=50, blit=True)
# writer = animation.Writers['ffmpeg'](fps=30, bitrate=1800)
anim.save('figure_gen/sloped_better_soln.mp4')

print('Done!')