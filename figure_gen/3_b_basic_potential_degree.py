from generaliser_compare import do_generalisation_tests_and_plots

TESTING_MAX_FOURIER_MODE = 2
TESTING_MIN_TIME = 0.25
TESTING_MAX_TIME = 0.75
NUM_TESTS = 100

DATA_DRIVEN_FOURIER_MODE_MODELS = [
    # (0, 'kiiara_2_results/models/31/1/model_at_epoch_1000.pt'),
    (2, 'kiiara_2_results/models/26/1/model_at_epoch_1000.pt'),
    (4, 'kiiara_2_results/models/27/1/model_at_epoch_1000.pt'),
    (6, 'kiiara_2_results/models/28/1/model_at_epoch_1000.pt'),
    (8, 'kiiara_2_results/models/29/1/model_at_epoch_1000.pt'),
    (10, 'kiiara_2_results/models/30/1/model_at_epoch_1000.pt')
]

PHYSICS_DRIVEN_FOURIER_MODE_MODELS = [
    
]

do_generalisation_tests_and_plots(
    max_test_fourier_mode=TESTING_MAX_FOURIER_MODE,
    min_test_time=TESTING_MIN_TIME,
    max_test_time=TESTING_MAX_TIME,
    num_tests=NUM_TESTS,
    supervised_models=DATA_DRIVEN_FOURIER_MODE_MODELS,
    unsupervised_models=PHYSICS_DRIVEN_FOURIER_MODE_MODELS,
    plot_title='Generalisation To Complicated Potentials',
    plot_xlabel='Max Degree of Potential',
    figure_save_location='figure_gen/3_b_basic_potential_degree'
)
