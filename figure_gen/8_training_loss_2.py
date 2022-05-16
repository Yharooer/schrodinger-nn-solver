import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas

TRAINING_LOSSES = 'figure_gen/the_training_losses/unsupervised_alternative_3/progress.csv'
CUTOFF = 1000

LOSS_DROPS = [189,932,968,978,992]

df = pandas.read_csv(TRAINING_LOSSES)
df = df[df['Epoch'] <= CUTOFF]

for font in mpl.font_manager.findSystemFonts('fonts'):
    mpl.font_manager.fontManager.addfont(font)

# Plot animations
mpl.rcParams['font.family'] = 'EB Garamond'
mpl.rcParams['font.size'] = 9
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble']="\\usepackage{mathpazo}"

fig = plt.figure(figsize=(4.0,4.5))

for d in LOSS_DROPS:
    plt.axvline(d,color='#808080', linewidth='1')

val_handle = plt.plot(df['Epoch'], df['Validation Loss']/10, color='#808080') 
dt_handle = plt.plot(df['Epoch'], df['Diff. Loss']/5, color='#00b35a')
ic_handle = plt.plot(df['Epoch'], df['IC Loss'], color='#ff0080')
bc_handle = plt.plot(df['Epoch'], df['BC Loss'], color='#ff8000')
tot_handle = plt.plot(df['Epoch'], df['Total Loss'], color='#0000ff')
 

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Physics-Driven Model Losses', fontsize=9, y=1.25)

leg = plt.legend([dt_handle[0], bc_handle[0], ic_handle[0], tot_handle[0], val_handle[0]], ['Differential Loss', 'Boundary Cond. Loss', 'Initial Cond. Loss', 'Total Loss', 'Validation Loss'], shadow=False, edgecolor='#000000', framealpha=1.0, ncol=2, bbox_to_anchor=(0,1.0,1,0.0), loc='lower left', mode='expand', prop={'size': 9})
leg.get_frame().set_boxstyle('Square', pad=0.1)
leg.get_frame().set(capstyle='butt', joinstyle='miter', linewidth=1.0)


plt.tight_layout()

plt.savefig('figure_gen/8_training_loss_2.pdf')
plt.savefig('figure_gen/8_training_loss_2.pgf')


plt.show()
