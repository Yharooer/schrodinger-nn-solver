# Schrodinger NN Solver by Bentley Carr

Part III Project for Part III Physics at the University of Cambridge.

## Examples
- Train a fully supervised model over t=\[0,0.5\] with initial states sampling from 3 fourier modes and no potential with each initial state evaluated at 10 linearly spaced points in time and 25 linearly spaced points in space.
```
python3 generate_training_data.py 0.5 --NUM_TIME_STEPS 10 --TRAINING_GRID_SIZE 25 --NUM_INITIAL_STATES 4000 --NUM_FOURIER_MODES 3 --NUM_POTENTIAL_DEGREE -1
python3 train_model.py <id of generated data> 1 0 0 0 0 0
python3 run_samples.py <path to model> 0.5
```

- Train a fully supervised model over t=\[0,0.5\] with initial states sampling from 3 fourier modes and linear potential with each initial state evaluated at 10 linearly spaced points in time and 25 linearly spaced points in space.
```
python3 generate_training_data.py 0.5 --NUM_TIME_STEPS 10 --TRAINING_GRID_SIZE 25 --NUM_INITIAL_STATES 4000 --NUM_FOURIER_MODES 3 --NUM_POTENTIAL_DEGREE 1
python3 train_model.py <id of generated data> 1 0 0 0 0 0
python3 run_samples.py <path to model> 0.5
```

- Train a fully supervised model over t=\[0,0.5\] with initial states sampling from 3 fourier modes and linear potential with each initial state evaluated at 10 randomised points in time and 25 randomised points in space.
```
python3 generate_training_data.py 0.5 --NUM_TIME_STEPS 10 --TRAINING_GRID_SIZE 25 --NUM_INITIAL_STATES 4000 --NUM_FOURIER_MODES 3 --NUM_POTENTIAL_DEGREE 1 --RANDOM_X_SAMPLING --RANDOM_T_SAMPLING
python3 train_model.py <id of generated data> 1 0 0 0 0 0
python3 run_samples.py <path to model> 0.5
```

- Find optimal hyperparameters for a fully unsupervised model over t=\[0,0.5\] with initial states sampling from 5 fourier modes and quartic potential with 1 000 000 initial states.
```
python3 generate_training_data.py 0.5 --NUM_TIME_STEPS 1 --TRAINING_GRID_SIZE 1 --NUM_INITIAL_STATES 1000000 --NUM_FOURIER_MODES 5 --NUM_POTENTIAL_DEGREE 4 --UNSUPERVISED
python3 train_hyperparameters.py
```
Will have to edit `train_hyperparameters.py` to point to the correct training data, set `HYPERPARAM_MSE` to zero, set the hyperparameter search range and set the keyword `UNSUPERVISED_RANDOMISED=True` in `default_params`.


- Train a fully unsupervised model over t=\[0,0.5\] with initial states sampling from 5 fourier modes and quartic potential with 1 000 000 initial states.
```
python3 generate_training_data.py 1.0 --NUM_TIME_STEPS 1 --TRAINING_GRID_SIZE 1 --NUM_INITIAL_STATES 1000000 --NUM_FOURIER_MODES 5 --NUM_POTENTIAL_DEGREE 4 --UNSUPERVISED
python3 train_model.py <id of generated data> 0 <HYPERPARAM_DT> <HYPERPARAM_BC> <HYPERPARAM_IC> <HYPERPARAM_NORM> <HYPERPARAM_ENERGY> --UNSUPERVISED_RANDOMISED --RAND_TIME 0.5 
python3 run_samples.py <path to model> 0.5
```