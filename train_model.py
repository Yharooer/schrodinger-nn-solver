import os
import argparse
import re
import torch
import pickle
import time
import json
import numpy as np
import pandas
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.schrodinger_losses import loss_function, get_loss_short_labels
from utils.model_definition import SchrodingerModel
from utils.argparser_helper import reduce_to_single_arguments, check_args_positive_numbers, print_arguments
from utils.dataset import SchrodingerDataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(training_path, validation_path):
    try:
        with open(training_path, 'rb') as file:
            training_state_dict = torch.load(file)
            training_data = SchrodingerDataset.from_state_dict(training_state_dict)
    except BaseException as e:
        raise e
        raise ValueError(f'Failed to load training data \'{training_path}\'.')

    if validation_path == None:
        validation_x = None
        validation_y = None
    else:
        try:
            with open(validation_path, 'rb') as file:
                validation_state_dict = torch.load(file)
                validation_x = validation_state_dict['validation_x']
                validation_y = validation_state_dict['validation_y']
        except:
            raise ValueError(f'Failed to load validation data \'{validation_path}\'.')

    return training_data, validation_x, validation_y


def update_at_epoch(epoch, total_losses, component_losses, validation_losses, progress_file, unsupervised_randomised=False):
    if unsupervised_randomised:
        print('Epoch #{} *** Training Loss: {:.3e}; Validation Loss: {:.3e}'.format((epoch+1),total_losses[-1], validation_losses[-1]))
    else:
        print('Epoch #{} *** Training Loss: {:.3e}; Validation Loss: {:.3e}; MSE Loss: {:.3e}'.format((epoch+1),total_losses[-1], validation_losses[-1], component_losses[0][-1]))

    progress_file.write(f'{(epoch+1)},{total_losses[-1]},{component_losses[0][-1]},{component_losses[1][-1]},{component_losses[2][-1]},{component_losses[3][-1]},{component_losses[4][-1]},{component_losses[5][-1]},{validation_losses}\n')
    progress_file.flush()

def get_model_save_path(epoch, output_directory):
    return os.path.join(output_directory, f'model_at_epoch_{epoch+1}.pt')


def update_at_major(epoch, total_losses, component_losses, validation_losses, output_directory, params, model, optimiser):
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'params': params,
        'total_losses': total_losses,
        'component_losses': component_losses,
        'validation_losses': validation_losses
    }, get_model_save_path(epoch, output_directory))

    # Plot losses
    plt.figure(figsize=(7,7))
    plt.plot(total_losses)
    plt.plot(component_losses[0] if not params['UNSUPERVISED_RANDOMISED'] else 0)
    plt.plot(component_losses[1])
    plt.plot(component_losses[2])
    plt.plot(component_losses[3])
    plt.plot(component_losses[4])
    plt.plot(component_losses[5])
    plt.plot(validation_losses)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Total', 'MSE', 'dt', 'BCs', 'ICs', 'Norm.', 'Energy', 'Validation'])
    plt.title(f'Losses at Epoch {(epoch+1)} with Loss {"{:.3e}".format(total_losses[-1])}')
    plt.savefig(os.path.join(output_directory, f'losses_at_epoch_{epoch+1}.pdf'))
    plt.close()


def train_model(params, output_directory, call_at_epoch=None):
    if call_at_epoch != None and not callable(call_at_epoch):
        raise ValueError('call_at_epoch needs to be a function.')

    CHECKPOINT_FREQUENCY = params['CHECKPOINT_FREQUENCY']

    if params['FROM_CHECKPOINT'] == None:
        progress_file = open(os.path.join(output_directory, 'progress.txt'), 'w')
        progress_file.write('Epoch,Total Loss,MSE Loss,Diff. Loss,BC Loss,IC Loss,Norm. Loss,Energy Loss,Validation Loss\n')

        model = SchrodingerModel(hidden_dim=params['HIDDEN_LAYER_SIZE']).to(device)
        optm = torch.optim.Adam(model.parameters(), lr = params['LEARNING_RATE'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optm)

        starting_epoch = 0

        total_losses = []
        component_losses = [[],[],[],[],[],[]]
        validation_losses = []

    else:
        progress_file = open(os.path.join(output_directory, 'progress.txt'), 'a')

        checkpoint_data = torch.load(params['FROM_CHECKPOINT'])

        params = checkpoint_data['params']

        model = SchrodingerModel(hidden_dim=params['HIDDEN_LAYER_SIZE']).to(device)
        model.load_state_dict(checkpoint_data['model_state_dict'])

        optm = torch.optim.Adam(model.parameters(), lr = params['LEARNING_RATE'])
        optm.load_state_dict(checkpoint_data['optimizer_state_dict'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optm)

        starting_epoch = checkpoint_data['epoch']

        total_losses = checkpoint_data['total_losses']
        component_losses = checkpoint_data['component_losses']
        validation_losses = validation_losses['validation_losses']


    training_data, validation_x, validation_y = load_data(params['TRAINING_DATA'], params['VALIDATION_DATA'])

    train_data_loader = torch.utils.data.DataLoader(training_data, batch_size=params['BATCH_SIZE'], shuffle=True)

    nepochs = params['MAX_EPOCHS']
    exit_criteria = None

    for epoch in range(starting_epoch, nepochs):
        epoch_loss = 0
        loss_components = [0,0,0,0,0,0]

        # Train using training data
        for x,y in train_data_loader:
            x = x.to(device)
            y = y.to(device)

            batch_size = x.shape[0]

            if params['UNSUPERVISED_RANDOMISED']:
                randomised_positions = torch.rand((batch_size,))
                randomised_times = torch.rand((batch_size,))*params['RAND_MAX_TIME']

                x = x.clone()
                x[:,0] = randomised_positions
                x[:,1] = randomised_times

            # Global phase transformation
            if params['TRAINING_MIXING']:
                mixing_angle = 2*np.pi*torch.rand(batch_size, 1).to(device)
                cos_m = torch.cos(mixing_angle)
                sin_m = torch.sin(mixing_angle)

                new_real = cos_m*x[:,2:102] + sin_m*x[:,102:202]
                new_imag = -sin_m*x[:,2:102] + cos_m*x[:,102:202]

                x[:,2:102] = new_real
                x[:,102:202] = new_imag

                new_y_real = cos_m[:,0]*y[:,0] + sin_m[:,0]*y[:,1]
                new_y_imag = -sin_m[:,0]*y[:,0] + cos_m[:,0]*y[:,1]

                y[:,0] = new_y_real
                y[:,1] = new_y_imag

            # Mixing between states
            if params['TRAINING_MIXING'] and params['UNSUPERVISED_RANDOMISED']:
                ## TODO can also do local phase transformations to add randomness

                # Linear combination of potentials
                mix_cffs = torch.normal(torch.zeros((batch_size,1)),1).to(device)
                x[:,202:] = torch.roll(x[:,202:], shifts=1, dims=0) + mix_cffs*torch.roll(x[:,202:], shifts=-1, dims=0)

                # Potential scaling
                if params['UNSUPERVISED_POTENTIAL_SCALING'] != 0:
                    scale_cffs = torch.normal(torch.zeros(batch_size, 1), params['UNSUPERVISED_POTENTIAL_SCALING']).to(device) + 1
                    x[:,202:] = scale_cffs * x[:,202:]

            optm.zero_grad()
            output = model(x)     
            loss, components = loss_function(output,x,y,model,params['HYPERPARAM_MSE'], params['HYPERPARAM_DT'], params['HYPERPARAM_BC'], params['HYPERPARAM_IC'], params['HYPERPARAM_NORM'], params['HYPERPARAM_ENERGY'], use_autograd=params['USE_AUTOGRAD'])
            
            loss.backward()
            optm.step()

            # Update training loss
            epoch_loss+=loss/len(train_data_loader)
            
            # Update losses of each loss component.
            for i in range(6):
                loss_components[i] += components[i]/len(train_data_loader)

        # Evaluate validation
        if validation_x != None and validation_y != None:
            with torch.no_grad():
                validation_loss = F.mse_loss(model(validation_x.to(device)), validation_y.to(device))
                scheduler.step(validation_loss)
        
        # Update running tally of losses
        total_losses.append(epoch_loss.detach().cpu().numpy())
        for i in range(6):
            if isinstance(loss_components[i], float):
                component_losses[i].append(loss_components[i])
            else:
                component_losses[i].append(loss_components[i].detach().cpu().numpy())
        if validation_x != None and validation_y != None:
            validation_losses.append(validation_loss.cpu().numpy())
        
        # Update progress
        update_at_epoch(epoch, total_losses, component_losses, validation_losses, progress_file, params['UNSUPERVISED_RANDOMISED'])

        # Call function at epoch (eg for use in Ray Tune)
        if call_at_epoch != None:
            call_at_epoch(epoch, total_losses, component_losses, validation_losses)

        # Update at checkpoint
        if CHECKPOINT_FREQUENCY != 0 and (epoch+1) % CHECKPOINT_FREQUENCY == 0:
            update_at_major(epoch, total_losses, component_losses, validation_losses, output_directory, params, model, optm)


    update_at_major(epoch, total_losses, component_losses, validation_losses, output_directory, params, model, optm)
    if exit_criteria == None:
        exit_criteria = "Max Epochs"

    return float(total_losses[-1]), [float(component_losses[i][-1]) for i in range(6)], float(validation_losses[-1]), exit_criteria


def get_arguments():
    parser = argparse.ArgumentParser(description="Creates training data.")
    parser.add_argument('TRAINING_DATA', type=str, nargs=1,
                        help='The training data to be used.')
    parser.add_argument('VALIDATION_DATA', type=str, nargs='?', default=None,
                        help='[Optional] The validation data to be used.')
    parser.add_argument('HYPERPARAM_MSE', type=float, nargs=1,
                        help='Hyperparameter for weighting of MSE to numerical solution in the loss function.')
    parser.add_argument('HYPERPARAM_DT', type=float, nargs=1,
                        help='Hyperparameter for differential term relating to Schrodinger\'s equation in the loss function.')
    parser.add_argument('HYPERPARAM_BC', type=float, nargs=1,
                        help='Hyperparameter for boundary conditions term in the loss function.')
    parser.add_argument('HYPERPARAM_IC', type=float, nargs=1,
                        help='Hyperparameter for initial conditions term in the loss function.')
    parser.add_argument('HYPERPARAM_NORM', type=float, nargs=1,
                        help='Hyperparameter for normalisation condition in the loss function.')
    parser.add_argument('HYPERPARAM_ENERGY', type=float, nargs=1,
                        help='Hyperparameter for conservation of energy in the loss function.')
    parser.add_argument('--LEARNING_RATE', type=float, nargs='?', default=0.001,
                        help='[Optional] Learning rate to be used with Adam optimiser. Defaults to 0.001.')
    parser.add_argument('--BATCH_SIZE', type=float, nargs='?', default=1000,
                        help='[Optional] Batch size to use during training. Defaults to 1000.')
    parser.add_argument('--MAX_EPOCHS', type=int, nargs='?', default=1000,
                        help='[Optional] Max number of epochs. Defaults to 1000.')
    parser.add_argument('--HIDDEN_LAYER_SIZE', type=float, nargs='?', default=500,
                        help='[Optional] Size of the two hidden layers. Defaults to 500.')
    parser.add_argument('--CHECKPOINT_FREQUENCY', type=int, nargs='?', default=10,
                        help='[Optional] Interval of checkpoints. For no checkpoints set to zero.')
    parser.add_argument('--FROM_CHECKPOINT', type=str, nargs='?', default=None,
                        help='Specify to continue training from a checkpoint.')
    parser.add_argument('--USE_AUTOGRAD', action='store_true',
                    help='Add this argument to use autograd. Warning: will use a lot of memory.')
    parser.add_argument('--UNSUPERVISED_RANDOMISED', action='store_true',
                        help='For unsupervised dataset, will randomise positions and times.')
    parser.add_argument('--RAND_MAX_TIME', type=float, nargs='?', default=0,
                        help='For unsupervised dataset, will set the maximum time.')
    parser.add_argument('--RAND_TIME_CUTOFF', type=float, nargs='?', default=0,
                        help='For unsupervised dataset, will set the length of the cutoff for time sampling.')
    parser.add_argument('--TRAINING_MIXING', action='store_true',
                        help='Will use global phase invariance to mix training data at each epoch. When combined with --UNSUPERVISED_RANDOMISED, will also take linear combinations of initial states and random combinations of potentials.')
    parser.add_argument('--UNSUPERVISED_POTENTIAL_SCALING', type=float, nargs='?', default=0,
                        help='Will scale the potential by a random number sampled with mean 1 standard deviation UNSUPERVISED_POTENTIAL_SCALING. A value greater than one will have the tendancy of making the potentials larger. If zero is provided, will not scale. Default is 0.')

    args = vars(parser.parse_args())
    reduce_to_single_arguments(args)
    check_arguments(args)
    print_arguments(args)
    return args


def check_arguments(args):
    if not os.path.isfile(args['TRAINING_DATA']):
        if os.path.isfile(os.path.join(args['TRAINING_DATA'], 'training_dataset.pt')):
            args['TRAINING_DATA'] = os.path.join(args['TRAINING_DATA'], 'training_dataset.pt')
        elif os.path.isfile(f'training_data/{args["TRAINING_DATA"]}/training_dataset.pt'):
            args['TRAINING_DATA'] = f'training_data/{args["TRAINING_DATA"]}/training_dataset.pt'
        else:
            raise ValueError(f'Cannot find training data file \'{args["TRAINING_DATA"]}\'.')

    if args['RAND_TIME_CUTOFF'] != 0:
        raise NotImplementedError('Unsupervised time cutoff not implemented yet.')

    if args['UNSUPERVISED_RANDOMISED'] and args['RAND_MAX_TIME'] <= 0:
        raise ValueError('When UNSUPERVISED_RANDOMISED is set, RAND_MAX_TIME must be set to some positive value.')

    if args['VALIDATION_DATA'] == None:
        typical_path = args['TRAINING_DATA'].replace('training_dataset.pt', 'validation_dataset.pt')
        if os.path.isfile(typical_path):
            args['VALIDATION_DATA'] = typical_path

    # TODO parsing checkpoint argument - in theory only need to specify an integer.

    check_args_positive_numbers(args, ['LEARNING_RATE', 'BATCH_SIZE', 'MAX_EPOCHS', 'HIDDEN_LAYER_SIZE'])

    if args['CHECKPOINT_FREQUENCY'] < 0:
        raise ValueError('CHECKPOINT_FREQUENCY must be a positive integer or zero.')


def extract_training_data_id(training_data_path):
    return re.sub("[^0-9]", "", training_data_path)


def get_output_folder(training_data_id):
    if not os.path.isdir('models'):
        print('Making directory \'models\'.')
        os.mkdir('models')

    if not os.path.isdir(f'models/{training_data_id}'):
        print(f'Making directory \'models/{training_data_id}\'.')
        os.mkdir(f'models/{training_data_id}')

    num = 1
    while True:
        directory = f'models/{training_data_id}/{num}'
        if not os.path.exists(directory):
            print(f'Making directory \'{directory}\'.')
            os.mkdir(directory)
            return directory
        num += 1


def get_root_directory(training_data_id):
    if not os.path.isdir('models'):
        print('Making directory \'models\'.')
        os.mkdir('models')

    if not os.path.isdir(f'models/{training_data_id}'):
        print(f'Making directory \'models/{training_data_id}\'.')
        os.mkdir(f'models/{training_data_id}')

    return f'models/{training_data_id}'


def create_params_file(args, directory):
    with open(f'{directory}/params.json', 'w') as params_file:
        params_file.write(json.dumps(args))


def main():
    params = get_arguments()

    # Create output directory
    training_data_id = extract_training_data_id(params['TRAINING_DATA'])
    output_dir = get_output_folder(training_data_id)
    create_params_file(params, output_dir)

    if device == "cuda":
        print(f"CUDA available. Using device \"{torch.cuda.get_device_name()}\".")
    else:
        print(f"Using CPU device.")

    # Train model
    print('Beginning training...')
    start = time.perf_counter()
    tot_loss, components_loss, validation_loss, exit_criteria = train_model(params, output_dir)
    end = time.perf_counter()
    print('Training finished.')

    print('Saving results.')

    # Update params
    params['TIME_TAKEN_SECONDS'] = end-start
    params['TRAINING_LOSS'] = tot_loss
    params['COMPONENTS_LOSS'] = components_loss
    params['VALIDATION_LOSS'] = validation_loss
    params['EXIT_CRITERIA'] = exit_criteria
    
    create_params_file(params, output_dir)

    # Update root results file.
    root_results_path = os.path.join(get_root_directory(training_data_id), 'results.csv')
    if os.path.isfile(root_results_path):
        results_df = pandas.read_csv()
    else:
        results_df = pandas.DataFrame()

    results_df = results_df.append(params, ignore_index=True)
    results_df.to_csv(root_results_path)

    print('Finished saving results.')


if __name__ == "__main__":
    main()
