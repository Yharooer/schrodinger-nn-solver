import numpy as np
import pandas 
import matplotlib.pyplot as plt

no_improv_path = 'performance/results_against_time_evolve_nn/25/particle_in_box_eigen1_data.csv'
mixing_only_path = 'performance/results_against_time_evolve_nn/26/particle_in_box_eigen1_data.csv'
random_only_path = 'performance/results_against_time_evolve_nn/27/particle_in_box_eigen1_data.csv'
both_improv_path = 'performance/results_against_time_evolve_nn/28/particle_in_box_eigen1_data.csv'

df_1 = pandas.read_csv(no_improv_path)
df_2 = pandas.read_csv(mixing_only_path)
df_3 = pandas.read_csv(random_only_path)
df_4 = pandas.read_csv(both_improv_path)

plt.plot(df_1['ts'], df_1['mse_error'], color='#ff0000')
plt.plot(df_2['ts'], df_2['mse_error'], color='#000088')
plt.plot(df_3['ts'], df_3['mse_error'], color='#00ff00')
plt.plot(df_4['ts'], df_4['mse_error'], color='#ff8800')
plt.legend(['no', 'mixing', 'random', 'both'])
plt.show()