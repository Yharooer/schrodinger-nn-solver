from generaliser_compare import do_generalisation_tests_and_plots

TESTING_MAX_FOURIER_MODE = 2
TESTING_MIN_TIME = 0.25
TESTING_MAX_TIME = 0.75
NUM_TESTS = 100

DATA_DRIVEN_FOURIER_MODE_MODELS = [
    (2, 'google_results/models/77/1/model_at_epoch_1000.pt'),
    (4, 'google_results/models/78/1/model_at_epoch_1000.pt'),
    (6, 'google_results/models/79/1/model_at_epoch_1000.pt'),
    (8, 'google_results/models/80/2/model_at_epoch_1000.pt'),
    (10, 'google_results/models/81/1/model_at_epoch_1000.pt')
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
    plot_title='Generalisation To More Initial States',
    plot_xlabel='Number of Fourier Modes',
    figure_save_location='figure_gen/3_a_basic_fourier_modes'
)
