import numpy as np
import torch
from utils.numerical_schrodinger import numerical_schrodinger
from utils.batch_interpolate import batch_interp
import scipy.interpolate
import random
from tqdm import tqdm

import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# TODO we should probably have a SchrodingerDatasetGenerator object which creates
# a SchrodingerDataset object cause right now we are passing way too many arguments
# into this when all we want to do is load the dataset.
class SchrodingerDataset(torch.utils.data.Dataset):
    def __init__(self, simulation_grid_size, training_grid_size, fourier_modes, potential_degree, max_time, ntimes, num_initials, random_x_sampling=None, random_t_sampling=None, potential_scale_factor=0, batch_time_eval_size=1000, unsupervised=None):
        self.simulation_grid_size = simulation_grid_size
        self.training_grid_size = training_grid_size
        self.fourier_modes = fourier_modes
        self.potential_degree = potential_degree
        self.max_time = max_time
        self.ntimes = ntimes
        self.num_initials = num_initials
        self.random_x_sampling = random_x_sampling
        self.random_t_sampling = random_t_sampling
        self.potential_scale_factor = potential_scale_factor
        self.batch_time_eval_size = batch_time_eval_size
        self.unsupervised = unsupervised

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

        print('Generating initial states.')
        for i in range(self.num_initials):
            psi0_real, psi0_imag, v = self._generate_initial()
            initials[0, :, i] = psi0_real.T
            initials[1, :, i] = psi0_imag.T
            initials[2, :, i] = v.T

        ts = np.linspace(0, self.max_time, self.ntimes)

        print('Done generating initial states. Time evolving.')

        if not self.unsupervised:

            # If random_t_sample then do in batches. Otherwise do all at once.
            if self.random_t_sampling:
                
                if self.ntimes > self.batch_time_eval_size:
                    print('WARNING: NUM_TIME_STEPS is larger than BATCH_TIME_EVAL_SIZE. We will do in batches in size NUM_TIME_STEPS.')
                
                num_initals_per_batch = int(np.max([1,np.floor(self.batch_time_eval_size / self.ntimes)]))
                num_batches = int(np.ceil(self.num_initials / num_initals_per_batch))
                
                integrated = np.zeros((2, self.simulation_grid_size, self.num_initials, self.ntimes))
                random_t_value = np.zeros((self.num_initials, self.ntimes))

                print('Starting numerical integration.')
                for i in tqdm(range(num_batches)):
                    start_index = i*num_initals_per_batch
                    end_index = np.min([self.num_initials,start_index+num_initals_per_batch])
                    
                    ts = np.random.rand((end_index - start_index)*self.ntimes)*self.max_time
                    sorted_ts = np.sort(ts)
                    argsort_ts = np.argsort(ts)

                    integrated_batch = numerical_schrodinger(initials[:,:,start_index:end_index], sorted_ts, self.simulation_grid_size, 1)

                    # I'm sorry if this is slow.
                    for j in range(end_index-start_index):
                        start_time_index = j*self.ntimes
                        time_indices = argsort_ts[start_time_index:start_time_index+self.ntimes]
                        integrated[:,:,i*num_initals_per_batch+j,:] = integrated_batch[:,:,j,time_indices]
                        random_t_value[i*num_initals_per_batch+j,:] = sorted_ts[time_indices]

                print('Finished numerical integration.')
            else:
                print('Starting numerical integration.')
                integrated = numerical_schrodinger(initials, ts, self.simulation_grid_size, 1)
                print('Finished numerical integration.')

            # Scale down target from simulation grid size to training grid size.
            if not self.random_x_sampling:
                xs_train_grid = np.linspace(0,1,self.training_grid_size)

                if self.training_grid_size != self.simulation_grid_size:
                    integrated = scipy.interpolate.interp1d(np.linspace(0,1,self.simulation_grid_size), integrated, kind='linear', axis=1)(xs_train_grid)

            if self.random_x_sampling:
                integrated_tmp = np.swapaxes(integrated, 1, 3)
                integrated_tmp = np.reshape(integrated_tmp, (2*self.num_initials*self.ntimes, self.simulation_grid_size))
                
                xs_train_grid = np.random.rand(self.training_grid_size, self.num_initials, self.ntimes)

                xs_train_grid_tmp = np.zeros((2, self.training_grid_size, self.num_initials, self.ntimes))
                xs_train_grid_tmp[0,:,:,:] = xs_train_grid
                xs_train_grid_tmp[1,:,:,:] = xs_train_grid
                
                xs_train_grid_reshape = np.reshape(np.swapaxes(xs_train_grid_tmp, 1, 3), (2*self.num_initials*self.ntimes, self.training_grid_size))

                integrated_tmp = batch_interp(torch.tensor(integrated_tmp).to(device), torch.tensor(xs_train_grid_reshape).to(device)).cpu().numpy()

                integrated_tmp = np.reshape(integrated_tmp, (2, self.ntimes, self.num_initials, self.training_grid_size))
                integrated = np.swapaxes(integrated_tmp, 1, 3)
        else:
            integrated = None

        print('Done time evolving.')

        # Scale down initial states from simulation grid size to training grid size.
        if self.simulation_grid_size != 100:
            initials = scipy.interpolate.interp1d(np.linspace(0,1,self.simulation_grid_size), initials, kind='linear', axis=1)(np.linspace(0,1,100))

        self.data = []

        print('Compiling dataset from numerical integration.')
        for i in tqdm(range(self.num_data)):
            a = i % self.ntimes                              # time index
            b = int(i/self.ntimes) % self.training_grid_size # space index
            c = int(i/self.ntimes/self.training_grid_size)   # psi0 index

            x_real = initials[0, :, c]
            x_imag = initials[1, :, c]
            x_potl = initials[2, :, c]
            
            if self.unsupervised:
                x_position = np.random.rand((1))[0]
            elif self.random_x_sampling:
                x_position = xs_train_grid[b,c,a]
            else:
                x_position = xs_train_grid[b]

            if self.random_t_sampling:
                t_value = random_t_value[c,a]
            else:
                t_value = ts[a]

            x = np.concatenate((np.array([x_position, t_value]), x_real, x_imag, x_potl))

            if self.unsupervised:
                y = np.array([0,0])
            else:
                y_real = integrated[0, b, c, a]
                y_imag = integrated[1, b, c, a]

                y = np.array([y_real, y_imag])

            x = torch.tensor(x).float()
            y = torch.tensor(y).float()

            self.data.append([x, y])

        print('Done compiling dataset.')

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
        scale_factor = np.sqrt(2/scale_factor)
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
        
            if self.potential_scale_factor > 0:
                scale_factor = np.random.normal(1, self.potential_scale_factor) + 1
                v_cffs = v_cffs * scale_factor

            v_cffs_mesh = np.outer(v_cffs, np.ones(self.simulation_grid_size))

        def potential_function(x):
            if self.potential_degree < 0:
                return 0*x

            x_shift = 2*x - 1
            ns = np.arange(0,self.potential_degree+1)
            xx, nn = np.meshgrid(x_shift, ns)
            return np.sum(v_cffs_mesh*np.power(xx,nn), axis=0)

        x = np.linspace(0, 1, self.simulation_grid_size)
        psi_real, psi_imag = init_wave_function(x)
        v = potential_function(x)

        return psi_real, psi_imag, v