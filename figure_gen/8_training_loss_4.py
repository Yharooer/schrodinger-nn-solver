import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas

TRAINING_LOSSES = 'figure_gen/the_training_losses/modified_unsupervised/progress_full.csv'
CONTROL_LOSSES = 'figure_gen/the_training_losses/modified_unsupervised_control/progress.csv'

LOSS_DROPS = [600,3590]

df = pandas.read_csv(TRAINING_LOSSES)
df2 = pandas.read_csv(CONTROL_LOSSES)

df2.loc[df2.index[-1], 'Epoch'] = df.loc[df.index[-1], 'Epoch']

for font in mpl.font_manager.findSystemFonts('fonts'):
    mpl.font_manager.fontManager.addfont(font)

# Plot animations
mpl.rcParams['font.family'] = 'EB Garamond'
mpl.rcParams['font.size'] = 9
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble']="\\usepackage{mathpazo}"

fig = plt.figure(figsize=(6.3,5.0))

for d in LOSS_DROPS:
    plt.axvline(d,color='#808080', linewidth='1')

# val_handle = plt.plot(df['Epoch'], df['Validation Loss']/10, color='#808080') 
mod_dt_handle = plt.plot(df['Epoch'], df['Diff. Loss'], color='#007d3f')
mod_ic_handle = plt.plot(df['Epoch'], df['IC Loss'], color='#b3005a')
mod_bc_handle = plt.plot(df['Epoch'], df['BC Loss'], color='#b35a00')
mod_tot_handle = plt.plot(df['Epoch'], df['Total Loss'], color='#0000b3')

ctrl_dt_handle = plt.plot(df2['Epoch'], df2['Diff. Loss'], color='#4dca8c')
ctrl_ic_handle = plt.plot(df2['Epoch'], df2['IC Loss'], color='#ff4da6')
ctrl_bc_handle = plt.plot(df2['Epoch'], df2['BC Loss'], color='#ffa64d')
ctrl_tot_handle = plt.plot(df2['Epoch'], df2['Total Loss'], color='#4d4dff')
 

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Modified Physics-Driven Model Losses', fontsize=9, y=1.20)

leg = plt.legend([mod_dt_handle[0], ctrl_dt_handle[0], mod_bc_handle[0], ctrl_bc_handle[0], mod_ic_handle[0], ctrl_ic_handle[0], mod_tot_handle[0], ctrl_tot_handle[0]], ['Mod. Diff. Loss', 'Ctrl. Diff. Loss', 'Mod. B.C. Loss', 'Ctrl. B.C. Loss', 'Mod. I.C. Loss', 'Ctrl. I.C. Loss', 'Mod. Total Loss', 'Ctrl. Total Loss'], shadow=False, edgecolor='#000000', framealpha=1.0, ncol=4, bbox_to_anchor=(0,1.0,1,0.0), loc='lower left', mode='expand', prop={'size': 9})
leg.get_frame().set_boxstyle('Square', pad=0.1)
leg.get_frame().set(capstyle='butt', joinstyle='miter', linewidth=1.0)


plt.tight_layout()

plt.savefig('figure_gen/8_training_loss_4.pdf')
plt.savefig('figure_gen/8_training_loss_4.pgf')


plt.show()
