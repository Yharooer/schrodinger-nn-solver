import numpy as np
import torch
from utils.numerical_schrodinger import numerical_schrodinger
import scipy.interpolate


import matplotlib.pyplot as plt

class SchrodingerDataset(torch.utils.data.Dataset):
    def __init__(self, simulation_grid_size, training_grid_size, fourier_modes, potential_degree, max_time, ntimes, num_initials):
        self.simulation_grid_size = simulation_grid_size
        self.training_grid_size = training_grid_size
        self.fourier_modes = fourier_modes
        self.potential_degree = potential_degree
        self.max_time = max_time
        self.ntimes = ntimes
        self.num_initials = num_initials
        self.num_data = num_initials*ntimes*training_grid_size
        self.data = []

    def get_state_dict(self):
        return {
            'training_grid_size': self.training_grid_size,
            'simulation_grid_size': self.simulation_grid_size,
            'fourier_modes': self.fourier_modes,
            'potential_degree': self.potential_degree,
            'max_time': self.max_time,
            'ntimes': self.ntimes,
            'num_initials': self.num_initials,
            'data': self.data
        }

    def from_state_dict(state_dict):
        dataset = SchrodingerDataset(state_dict['simulation_grid_size'], state_dict['training_grid_size'], state_dict['fourier_modes'], state_dict['potential_degree'] if 'potential_degree' in state_dict else 0, state_dict['max_time'], state_dict['ntimes'], state_dict['num_initials'])
        dataset.data = state_dict['data']
        return dataset

    def initialise(self):
        initials = np.empty((3, self.simulation_grid_size, self.num_initials))

        xs = np.linspace(0, 1, self.simulation_grid_size)

        for i in range(self.num_initials):
            psi0_real, psi0_imag, v = self._generate_initial()
            initials[0, :, i] = psi0_real.T
            initials[1, :, i] = psi0_imag.T
            initials[2, :, i] = v.T

        ts = np.linspace(0, self.max_time, self.ntimes)
        integrated = numerical_schrodinger(initials, ts, self.simulation_grid_size, 1)

        # Scale down from simulation grid size to training grid size.
        if self.training_grid_size != self.simulation_grid_size:
            integrated = scipy.interpolate.interp1d(np.linspace(0,1,self.simulation_grid_size), integrated, kind='linear', axis=1)(np.linspace(0,1,self.training_grid_size))

            plt.plot(np.linspace(0,1,self.training_grid_size), integrated[0,:,0,0])
            plt.plot(np.linspace(0,1,self.training_grid_size), integrated[1,:,0,0])

        self.data = []

        xs_train_grid = np.linspace(0,1,self.training_grid_size)

        for i in range(self.num_data):
            a = i % self.ntimes                 # time index
            b = int(i/self.ntimes) % self.training_grid_size  # space index
            c = int(i/self.ntimes/self.training_grid_size)  # psi0 index

            x_real = initials[0, :, c]
            x_imag = initials[1, :, c]
            x_potl = initials[2, :, c]

            x = np.concatenate((np.array([xs_train_grid[b], ts[a]]), x_real, x_imag, x_potl))

            y_real = integrated[0, b, c, a]
            y_imag = integrated[1, b, c, a]

            y = np.array([y_real, y_imag])

            x = torch.tensor(x).float()
            y = torch.tensor(y).float()

            self.data.append([x, y])

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        return self.data[index]

    def _generate_initial(self):
        # create the initial wave function
        fourier_real_coefficients = 2*np.random.rand(self.fourier_modes)-1
        fourier_imag_coefficients = 2*np.random.rand(self.fourier_modes)-1
        n = np.arange(start=1, stop=self.fourier_modes+1, step=1)

        scale_factor = np.sum(fourier_real_coefficients**2) + np.sum(fourier_imag_coefficients**2)
        scale_factor = (2*scale_factor)**0.5
        fourier_real_coefficients *= scale_factor
        fourier_imag_coefficients *= scale_factor

        def init_wave_function(x):
            x = np.pi*x
            psi_real = np.sin(np.outer(x, n))
            psi_real = psi_real*fourier_real_coefficients
            psi_real = np.sum(psi_real, axis=-1)

            psi_imag = np.sin(np.outer(x, n))
            psi_imag = psi_imag*fourier_imag_coefficients
            psi_imag = np.sum(psi_imag, axis=-1)

            return psi_real, psi_imag

        if self.potential_degree >= 0:
            v_cffs = np.random.normal(size=(self.potential_degree+1))
            v_cffs_mesh = np.outer(v_cffs, np.ones(100))

        def potential_function(x):
            if self.potential_degree < 0:
                return 0*x

            x_shift = 2*x - 1
            ns = np.arange(0,self.potential_degree+1)
            xx, nn = np.meshgrid(x_shift, ns)
            return np.sum(v_cffs_mesh*np.power(xx,nn), axis=0)

        x = np.linspace(0, 1, 100)
        psi_real, psi_imag = init_wave_function(x)
        v = potential_function(x)

        return psi_real, psi_imag, v