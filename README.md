```
python3 generate_training_data.py 0.5 --NUM_TIME_STEPS 20 --NUM_FOURIER_MODES 3 --TRAINING_GRID_SIZE 25 --NUM_INITIAL_STATES 1000
python3 train_model.py 16 1 0.1 0.025 0.025 0.025 0.025
```