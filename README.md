# Physics-Driven Modelling of the Schrödinger Equation

Quantum mechanics is governed by the Schrödinger equation which has few analytical solutions and can only be solved generally with slow error-prone numerical methods. We detail two machine-learning models which are able to predict solutions to the 1D Schrödinger equation for a particle confined to a box under the influence of an arbitrary potential. We present the data-driven model which is trained from numerical solutions and the physics-driven model which discovers solutions on its own during training with only knowledge of the governing PDE and boundary conditions. We find that both methods are able to reproduce solutions to the Schrödinger equation to a good approximation. The physics-driven model demonstrates better performance than the data-driven model for simple input vectors but the data-driven model is better at generalising to more complicated initial state and potentials. In particular the physics-driven model struggles learning fast dynamics of the Schrödinger equation. We compare the physics-driven model to the numerical Runge–Kutta method and we demonstrate that our model is able to predict solutions 1125 times faster than the numerical method while demonstrating similar performance.

[Read the report.](bentley_schrod_report_public.pdf)

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