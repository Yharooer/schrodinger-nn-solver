from generaliser_compare import do_generalisation_tests_and_plots

TESTING_MAX_FOURIER_MODE = 2
TESTING_MIN_TIME = 0.25
TESTING_MAX_TIME = 0.75
NUM_TESTS = 100

# TODO we need to convert these to RMS potentials
DATA_DRIVEN_FOURIER_MODE_MODELS = [
    (1.5, 'google_results/models/82/1/model_at_epoch_1000.pt'), # Potential scaling 0


    # (0, 'kiiara_2_results/models/43/1/model_at_epoch_1000.pt'),

    (2.9, 'google_results/models/83/1/model_at_epoch_1000.pt'), # Potential scaling 1
    (4.4, 'google_results/models/84/1/model_at_epoch_1000.pt'), # Potential scaling 2
    (5.6, 'google_results/models/85/1/model_at_epoch_1000.pt'), # Potential scaling 3
    (6.8, 'google_results/models/86/1/model_at_epoch_1000.pt'), # Potential scaling 4
    (8.4, 'google_results/models/87/1/model_at_epoch_1000.pt'), # Potential scaling 5
    (9.9, 'google_results/models/88/1/model_at_epoch_1000.pt'), # Potential scaling 6
    (11.6, 'google_results/models/89/1/model_at_epoch_1000.pt'), # Potential scaling 7

    # (10, 'kiiara_2_results/models/53/1/model_at_epoch_1000.pt'),
    # (13, 'kiiara_2_results/models/54/1/model_at_epoch_1000.pt'),
    # (16, 'kiiara_2_results/models/55/1/model_at_epoch_1000.pt'),
    # (18, 'kiiara_2_results/models/56/1/model_at_epoch_1000.pt'),


    # (20, 'kiiara_2_results/models/49/1/model_at_epoch_1000.pt'),
    # (40, 'kiiara_2_results/models/50/1/model_at_epoch_1000.pt'),
    # (60, 'kiiara_2_results/models/51/1/model_at_epoch_1000.pt'),
    # (80, 'kiiara_2_results/models/47/1/model_at_epoch_1000.pt'),
    # (100, 'kiiara_2_results/models/52/1/model_at_epoch_1000.pt')
]

PHYSICS_DRIVEN_FOURIER_MODE_MODELS = [
    #(0, 'google_results/models/94/1/model_at_epoch_810.pt'),
    (1.9, 'google_results/models/95/6/model_at_epoch_1000.pt'),  # Scaling factor 0
    (4.9, 'google_results/models/94/2/model_at_epoch_670.pt'),  # Scaling factor 2
    (8.8, 'google_results/models/94/3/model_at_epoch_670.pt'), # Scaling factor 4
    (13.5, 'google_results/models/94/4/model_at_epoch_670.pt'), # Scaling factor 6
    # (8, 'google_results/models/94/5/model_at_epoch_760.pt'),

    # (10, 'google_results/models/94/11/model_at_epoch_240.pt'),
    # (13, 'google_results/models/94/12/model_at_epoch_230.pt'),
    # (16, 'google_results/models/94/13/model_at_epoch_240.pt'),
    # (18, 'google_results/models/94/14/model_at_epoch_240.pt'),

    #(20, 'google_results/models/95/1/model_at_epoch_260.pt'),
    # (40, 'google_results/models/95/2/model_at_epoch_260.pt'),
    # (60, 'google_results/models/95/3/model_at_epoch_1000.pt'),
    # (80, 'google_results/models/95/4/model_at_epoch_260.pt'),
    # (100, 'google_results/models/95/5/model_at_epoch_260.pt')

]

do_generalisation_tests_and_plots(
    max_test_fourier_mode=TESTING_MAX_FOURIER_MODE,
    min_test_time=TESTING_MIN_TIME,
    max_test_time=TESTING_MAX_TIME,
    num_tests=NUM_TESTS,
    supervised_models=DATA_DRIVEN_FOURIER_MODE_MODELS,
    unsupervised_models=PHYSICS_DRIVEN_FOURIER_MODE_MODELS,
    plot_title='Generalisation To Larger Potentials',
    plot_xlabel='RMS Potential Strength',
    figure_save_location='figure_gen/3_c_basic_potential_degree',
    include_zero_benchmark=True
)
