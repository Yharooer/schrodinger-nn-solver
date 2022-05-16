import os
import argparse
import re
import torch
import pickle
import time
import json
import numpy as np
import pandas
import sys
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

    progress_file.write(f'{(epoch+1)},{total_losses[-1]},{component_losses[0][-1]},{component_losses[1][-1]},{component_losses[2][-1]},{component_losses[3][-1]},{component_losses[4][-1]},{component_losses[5][-1]},{validation_losses[-1]}\n')
    progress_file.flush()

def get_model_save_path(epoch, output_directory):
    return os.path.join(output_directory, f'model_at_epoch_{epoch+1}.pt')


def update_at_major(epoch, total_losses, component_losses, validation_losses, learning_rate_drops, output_directory, params, model, optimiser, scheduler, ray_tune_checkpoint_dir_func=None):
    if ray_tune_checkpoint_dir_func == None:
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler != None else None,
            'params': params,
            'total_losses': total_losses,
            'component_losses': component_losses,
            'validation_losses': validation_losses,
            'output_directory': output_directory,
            'learning_rate_drops': learning_rate_drops
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

        # Plot learning rate drops
        for lrd in learning_rate_drops:
            eph, lr = lrd
            if eph != 0:
                plt.axvline(eph, color='#aaaaaa', lw=0.5)

        plt.legend(['Total', 'MSE', 'dt', 'BCs', 'ICs', 'Norm.', 'Energy', 'Validation'])        
        plt.title(f'Losses at Epoch {(epoch+1)} with Loss {"{:.3e}".format(total_losses[-1])}')
        plt.savefig(os.path.join(output_directory, f'losses_at_epoch_{epoch+1}.pdf'))
        plt.close()
    else:
        with ray_tune_checkpoint_dir_func(epoch) as checkpoint_dir:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'params': params,
                'total_losses': total_losses,
                'component_losses': component_losses,
                'validation_losses': validation_losses,
                'output_directory': output_directory,
                'learning_rate_drops': learning_rate_drops
            }, os.path.join(checkpoint_dir, 'model.pt'))


def get_current_lr(optm):
    lrs = list(map(lambda param_group: float(param_group['lr']), optm.param_groups))
    return np.mean(lrs)


def train_model(params, output_directory, call_at_epoch=None, ray_tune_checkpoint_dir_func=None):
    if call_at_epoch != None and not callable(call_at_epoch):
        raise ValueError('call_at_epoch needs to be a function.')

    if params['FROM_CHECKPOINT'] == None:
        progress_file = open(os.path.join(output_directory, 'progress.csv'), 'w')
        progress_file.write('Epoch,Total Loss,MSE Loss,Diff. Loss,BC Loss,IC Loss,Norm. Loss,Energy Loss,Validation Loss\n')

        model = SchrodingerModel(hidden_dim=params['HIDDEN_LAYER_SIZE'], num_layers=params['NUM_HIDDEN_LAYERS']).to(device)
        optm = torch.optim.Adam(model.parameters(), lr = params['LEARNING_RATE'])

        if params['NO_REDUCE_LR']:
            scheduler = None
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optm, min_lr=params['MINIMUM_LEARNING_RATE'], verbose=True)

        starting_epoch = 0

        learning_rate_drops = [(0, params['LEARNING_RATE'])]
        
        total_losses = []
        component_losses = [[],[],[],[],[],[]]
        validation_losses = []

    else:
        progress_file = open(os.path.join(output_directory, 'progress.csv'), 'a')

        checkpoint_data = torch.load(params['FROM_CHECKPOINT'])

        model = SchrodingerModel(hidden_dim=params['HIDDEN_LAYER_SIZE'], num_layers=params['NUM_HIDDEN_LAYERS']).to(device)
        model.load_state_dict(checkpoint_data['model_state_dict'])

        if params['RESET_OPTIMISER']:
            optm = torch.optim.Adam(model.parameters(), lr = params['LEARNING_RATE'])
            if params['NO_REDUCE_LR']:
                scheduler = None
            else:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optm, min_lr=params['MINIMUM_LEARNING_RATE'], verbose=True)
        else:
            optm = torch.optim.Adam(model.parameters(), lr = params['LEARNING_RATE'])
            optm.load_state_dict(checkpoint_data['optimizer_state_dict'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optm)
            if checkpoint_data['scheduler_state_dict'] != None:
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])

            # Override the minimum learning rate in case it was changed in the arguments.
            scheduler.min_lrs = [params['MINIMUM_LEARNING_RATE']*len(optm.param_groups)]

        starting_epoch = checkpoint_data['epoch'] + 1

        if 'learning_rate_drops' in checkpoint_data:
            learning_rate_drops = checkpoint_data['learning_rate_drops']
        else:
            print('WARNING: \'learning_rate_drops\' variable not found. Continuing but lr drops might contain missing results.')
            learning_rate_drops = [(starting_epoch, get_current_lr(optm))]

        total_losses = checkpoint_data['total_losses']
        component_losses = checkpoint_data['component_losses']
        validation_losses = checkpoint_data['validation_losses']

        print(f'Loading from checkpoint at {starting_epoch}.')
        if params['UNSUPERVISED_RANDOMISED']:
            print('Epoch #{} *** Training Loss: {:.3e}; Validation Loss: {:.3e}'.format((starting_epoch),total_losses[-1], validation_losses[-1]))
        else:
            print('Epoch #{} *** Training Loss: {:.3e}; Validation Loss: {:.3e}; MSE Loss: {:.3e}'.format((starting_epoch),total_losses[-1], validation_losses[-1], component_losses[0][-1]))

    CHECKPOINT_FREQUENCY = params['CHECKPOINT_FREQUENCY']

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
                current_max_time = params['RAND_MAX_TIME'] + params['UNSUPERVISED_TIME_INCREASE_RATE']*epoch

                randomised_positions = torch.rand((batch_size,))
                randomised_times = torch.rand((batch_size,))*current_max_time

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
                if params['UNSUPERVISED_POTENTIAL_SCALING'] > 0:
                    scale_cffs = torch.normal(torch.zeros(batch_size, 1), params['UNSUPERVISED_POTENTIAL_SCALING']).to(device) + 1
                    x[:,202:] = scale_cffs * x[:,202:]
                if params['UNSUPERVISED_POTENTIAL_SCALING'] < 0:
                    x[:,202:] = 0

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
        validation_loss = None
        if validation_x != None and validation_y != None:
            with torch.no_grad():
                validation_loss = F.mse_loss(model(validation_x.to(device)), validation_y.to(device))
        
        # Use ReduceLROnPlateau scheduler
        if scheduler != None:
            if params['DYNAMIC_LR_USE_TRAINING_LOSS']:
                scheduler.step(epoch_loss)
            else:
                scheduler.step(validation_loss)

        # See whether the learning rate dropped to track dynamic LR
        old_lr = learning_rate_drops[-1][1]
        curr_lr = get_current_lr(optm)
        if curr_lr != old_lr:
            learning_rate_drops.append((epoch, curr_lr))
        
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
            update_at_major(epoch, total_losses, component_losses, validation_losses, learning_rate_drops, output_directory, params, model, optm, scheduler)


    update_at_major(epoch, total_losses, component_losses, validation_losses, output_directory, params, model, optm, scheduler, ray_tune_checkpoint_dir_func)
    if exit_criteria == None:
        exit_criteria = "Max Epochs"

    return float(total_losses[-1]), [float(component_losses[i][-1]) for i in range(6)], float(validation_losses[-1]), exit_criteria


def get_arguments():
    if '--RESUME_FROM_CHECKPOINT' in sys.argv:
        parser = argparse.ArgumentParser(description="Trains model from checkpoint with the same parameters.")
        parser.add_argument('--RESUME_FROM_CHECKPOINT', type=str, nargs='?', default=None,
                        help='Specify to continue training from a checkpoint with the same parameters. Other arguments cannot be present if this is used.')
        args = vars(parser.parse_args())
        reduce_to_single_arguments(args)
        print_arguments(args)
        return args

    parser = argparse.ArgumentParser(description="Trains model from training data.")

    # Required arguments
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
    
    # Arguments concerning training hyperparameters.
    parser.add_argument('--LEARNING_RATE', type=float, nargs='?', default=0.001,
                        help='[Optional] Learning rate to be used with Adam optimiser. Defaults to 0.001.')
    parser.add_argument('--BATCH_SIZE', type=float, nargs='?', default=1000,
                        help='[Optional] Batch size to use during training. Defaults to 1000.')
    parser.add_argument('--MAX_EPOCHS', type=int, nargs='?', default=1000,
                        help='[Optional] Max number of epochs. Defaults to 1000.')

    # Arguments concerning ReduceLROnPlateau and the optimiser.
    parser.add_argument('--DYNAMIC_LR_USE_TRAINING_LOSS', action='store_true',
                        help='Will use training loss instead of validation loss for ReduceLROnPlateau. Allows us to train data when the validation domain doesn\'t match the training domain.')
    parser.add_argument('--RESET_OPTIMISER', action='store_true',
                        help='Will reset the state of the optimiser and the scheduler.')
    parser.add_argument('--NO_REDUCE_LR', action='store_true',
                        help='Will not used ReduceLROnPlateau.')
    parser.add_argument('--MINIMUM_LEARNING_RATE', type=float, nargs='?', default=0,
                        help='The minimum learning rate which ReduceLROnPlateau will use.')

    # Arguments concerning the model hyperparameters.
    parser.add_argument('--HIDDEN_LAYER_SIZE', type=int, nargs='?', default=500,
                        help='[Optional] Size of each hidden layer. Defaults to 500.')
    parser.add_argument('--NUM_HIDDEN_LAYERS', type=int, nargs='?', default=2,
                        help='[Optional] Number of hidden layers. Defaults to 2.')
    
    # Arguments concerning checkpointing.
    parser.add_argument('--CHECKPOINT_FREQUENCY', type=int, nargs='?', default=10,
                        help='[Optional] Interval of checkpoints. For no checkpoints set to zero.')
    parser.add_argument('--FROM_CHECKPOINT', type=str, nargs='?', default=None,
                        help='Specify to continue training from a checkpoint with different training parameters. All arguments must be respecified. Allows for piecewise training of with different parameters.')
    parser.add_argument('--RESUME_FROM_CHECKPOINT', type=str, nargs='?', default=None,
                        help='Specify to continue training from a checkpoint with the same parameters. Must be specified on its own; all other arguments are ignored when this is specified.')

    # Arguments concerning randomisation during training.
    parser.add_argument('--UNSUPERVISED_RANDOMISED', action='store_true',
                        help='For unsupervised dataset, will randomise positions and times.')
    parser.add_argument('--RAND_MAX_TIME', type=float, nargs='?', default=0,
                        help='For unsupervised dataset, will set the maximum time. Combine with UNSUPERVISED_TIME_INCREASE_RATE to increase the max time each epoch.')
    parser.add_argument('--RAND_TIME_CUTOFF', type=float, nargs='?', default=0, 
                        help='For unsupervised dataset, will set the length of the cutoff for time sampling.')
    parser.add_argument('--TRAINING_MIXING', action='store_true',
                        help='Will use global phase invariance to mix training data at each epoch. When combined with --UNSUPERVISED_RANDOMISED, will also take linear combinations of initial states and random combinations of potentials.')
    parser.add_argument('--UNSUPERVISED_POTENTIAL_SCALING', type=float, nargs='?', default=0,
                        help='Will scale the potential by a random number sampled with mean 1 standard deviation UNSUPERVISED_POTENTIAL_SCALING. A value greater than one will have the tendancy of making the potentials larger. If zero is provided, will not scale. If a negative number is provided, will turn off potential. Default is 0.')
    parser.add_argument('--UNSUPERVISED_TIME_INCREASE_RATE', type=float, nargs='?', default=0,
                        help='The rate at which the RAND_MAX_TIME attribute will increase by each epoch. For example to increase by 0.1 each 100 epochs set to 1e-3. Defaults to zero.')

    # Argument concerning method of calculating derivatives.
    parser.add_argument('--USE_AUTOGRAD', action='store_true',
                    help='Add this argument to use autograd. May use a lot of memory.')
                    
    
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

    check_args_positive_numbers(args, ['LEARNING_RATE', 'BATCH_SIZE', 'MAX_EPOCHS', 'HIDDEN_LAYER_SIZE', 'NUM_HIDDEN_LAYERS'])

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

    # Check if FROM_CHECKPOINT is specified if we are given conflicting HIDDEN_LAYER_SIZE or NUM_HIDDEN_LAYERS.
    # Check if RESET_OPTIMISER is not set is there are differences in the learning rates.
    if 'FROM_CHECKPOINT' in params and params['FROM_CHECKPOINT'] != None:
        checkpoint_path = params['FROM_CHECKPOINT']
        checkpoint_data = torch.load(checkpoint_path, map_location=torch.device(device))
        checkpoint_params = checkpoint_data['params']

        if checkpoint_params['HIDDEN_LAYER_SIZE'] != params['HIDDEN_LAYER_SIZE']:
            print(f"WARNING: The argument HIDDEN_LAYER_SIZE specified ({params['HIDDEN_LAYER_SIZE']}) does not match the argument HIDDEN_LAYER_SIZE of the checkpoint data ({checkpoint_params['HIDDEN_LAYER_SIZE']}). We continue using the value specified by the checkpoint data.")
            params['HIDDEN_LAYER_SIZE'] = checkpoint_params['HIDDEN_LAYER_SIZE']
        
        if checkpoint_params['NUM_HIDDEN_LAYERS'] != params['NUM_HIDDEN_LAYERS']:
            print(f"WARNING: The argument NUM_HIDDEN_LAYERS specified ({params['NUM_HIDDEN_LAYERS']}) does not match the argument NUM_HIDDEN_LAYERS of the checkpoint data ({checkpoint_params['NUM_HIDDEN_LAYERS']}). We continue using the value specified by the checkpoint data.")
            params['NUM_HIDDEN_LAYERS'] = checkpoint_params['NUM_HIDDEN_LAYERS']

        if not params['RESET_OPTIMISER']:
            if checkpoint_params['LEARNING_RATE'] != params['LEARNING_RATE']:
                print(f"WARNING: The argument RESET_OPTIMISER was not provided and the argument LEARNING_RATE differs from the checkpoint data and the command line arguments. This may result in unintended behaviour such as the learning rate not changing.")
            if 'NO_REDUCE_LR' in checkpoint_params and checkpoint_params['NO_REDUCE_LR'] != params['NO_REDUCE_LR']:
                print(f"WARNING: The argument RESET_OPTIMISER was not provided and the argument NO_REDUCE_LR differs between the checkpoint data and the command line arguments. This may result in unintended behaviour of ReduceLROnPlateau.")

        print('Continuing with new parameters from checkpoint. The parameters we will use are:')
        print_arguments(params)

    # Create output directory
    if params['RESUME_FROM_CHECKPOINT'] == None:
        training_data_id = extract_training_data_id(params['TRAINING_DATA'])
        output_dir = get_output_folder(training_data_id)
        create_params_file(params, output_dir)
    else:
        checkpoint_path = params['RESUME_FROM_CHECKPOINT']
        checkpoint_data = torch.load(checkpoint_path, map_location=torch.device(device))
        params = checkpoint_data['params']
        output_dir = checkpoint_data['output_directory']
        print('Loaded input arguments from checkpoint.')
        params['RESUME_FROM_CHECKPOINT'] = checkpoint_path
        params['FROM_CHECKPOINT'] = checkpoint_path
        print_arguments(params)

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
