import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from _colors import create_palette, show_palette

EXP_FOLDER = os.path.dirname(os.path.abspath(__file__))
EXP_NAME = 'Timing'
filename = f'{EXP_NAME}_Results.json'
with open(os.path.join(EXP_FOLDER, filename), 'r') as f:
    results = json.load(f)


gates_names = ['AND','OR','XOR']
gates_names = ['XOR']

palette = create_palette(5)
show_palette(palette, save=True, name='palette')
color_D = palette[1]
color_DW = palette[3]

fig, axs = plt.subplots(1, 2, figsize=(5, 2*len(gates_names)), sharex=True,)

for i, gate in enumerate(gates_names):  
    results_gate = {k:v for k,v in results.items() if gate in k}
    H = [int(k.split('_')[-1]) for k in results_gate.keys()]
    H = sorted(list(set(H)))
    H = [2]
    
    def get_errors(gate, prunning_mode):
        results = {k:v for k,v in results_gate.items() if prunning_mode in k}                
        for h in H:
            key = f'{gate}_{prunning_mode}_{h}'
            print(key)
            errors = np.array(results[key]['mean_errors'])
            edges = np.array(results[key]['edges'])
            yield errors, edges

    def process_errors_and_edges(gate, label):
        errors, edges = map(np.array, zip(*get_errors(gate, label)))
        mean_errors = np.mean(errors, axis=1)
        mean_edges = np.mean(edges, axis=1)
        
        initial_edges = mean_edges[:, 0]
        mean_edges_percentual = 100 - 100 * mean_edges / initial_edges[:, None]
        mean_edges_percentual = mean_edges_percentual[:, :-1]
        
        return mean_errors, mean_edges_percentual, np.std(errors, axis=1), np.std(mean_edges_percentual, axis=1)

    mean_errors_D, mean_edges_D_percentual, std_errors_D, std_edges_D = process_errors_and_edges(gate, 'D')
    mean_errors_Dw, mean_edges_Dw_percentual, std_errors_Dw, std_edges_Dw = process_errors_and_edges(gate, 'Dw')


    #smooth the means with a window of 10
    window = 10
    mean_errors_D = np.array([np.convolve(e, np.ones(window)/window, mode='valid') for e in mean_errors_D])
    mean_errors_Dw = np.array([np.convolve(e, np.ones(window)/window, mode='valid') for e in mean_errors_Dw])
    #same with std
    std_errors_D = np.array([np.convolve(e, np.ones(window)/window, mode='valid') for e in std_errors_D])
    std_errors_Dw = np.array([np.convolve(e, np.ones(window)/window, mode='valid') for e in std_errors_Dw])

    Xerror = np.arange(mean_errors_D.shape[1])
    Xedges = np.arange(mean_edges_D_percentual.shape[1])
    # Plotting
    axs[0].plot(Xerror, mean_errors_D.flatten(), label='$D$', color=color_D, alpha=1)
    axs[0].plot(Xerror, mean_errors_Dw.flatten(), label='$D{w}$', color=color_DW, alpha=1)
    axs[1].plot(Xedges, mean_edges_D_percentual.flatten(), label='$D$', color=color_D, alpha=1)
    axs[1].plot(Xedges, mean_edges_Dw_percentual.flatten(), label='$D_{w}$', color=color_DW, alpha=1)

    # Plotting std bands
    axs[0].fill_between(Xerror, mean_errors_D.flatten()-std_errors_D.flatten(), mean_errors_D.flatten()+std_errors_D.flatten(), color=color_D, alpha=0.2)
    axs[0].fill_between(Xerror, mean_errors_Dw.flatten()-std_errors_Dw.flatten(), mean_errors_Dw.flatten()+std_errors_Dw.flatten(), color=color_DW, alpha=0.2)
    axs[1].fill_between(Xedges, mean_edges_D_percentual.flatten()-std_edges_D.flatten(), mean_edges_D_percentual.flatten()+std_edges_D.flatten(), color=color_D, alpha=0.2)
    axs[1].fill_between(Xedges, mean_edges_Dw_percentual.flatten()-std_edges_Dw.flatten(), mean_edges_Dw_percentual.flatten()+std_edges_Dw.flatten(), color=color_DW, alpha=0.2)

    axs[0].set_title('Error')
    axs[1].set_title('Edges Removed (%)')
    axs[0].set_xlabel('Training Steps')
    axs[1].set_xlabel('Training Steps')
    #legend
    axs[0].legend(fontsize='small')
    axs[1].legend(fontsize='small')
#for i, gn in enumerate(gates_names):
#    #axs[i,0].set_title(gn)
#    axs[i,0].set_ylabel(f'{gn}')
#    axs[i,1].set_ylabel(f'Absolute')
#    axs[i,0].set_xticks([]) # remove ticks
#    axs[i,1].set_xticks([]) # remove ticks
#    #axs[i,-1].set_xlabel('Hidden Nodes')
#
##axs[0,0].set_ylabel('Error')
##axs[0,1].set_ylabel('Edges Removed')
#axs[0,0].set_title('Error')
#axs[0,1].set_title('Edges Removed')
#
#axs[-1,0].set_xlabel('Hidden Nodes')
##add ticks
#axs[-1,0].set_xticks(H)
#axs[-1,1].set_xlabel('Hidden Nodes')
#axs[-1,1].set_xticks(H)
#
#axs[0,0].legend(fontsize='small')
## Gather handles and labels from both axes
#lines_1, labels_1 = axs[i, 1].get_legend_handles_labels()
#lines_2, labels_2 = ax2.get_legend_handles_labels()
## Combine handles and labels
#lines = lines_1 + lines_2
#labels = labels_1 + labels_2
## Create a single legend
#axs[-1, 1].legend(lines, labels, framealpha=0.5, fontsize='small', loc='lower right', ncol=2)
#
#
plt.subplots_adjust(hspace=0.5)

plt.tight_layout()


filename = 'Timing_Results.png'
plt.savefig(os.path.join(EXP_FOLDER,filename), dpi=600, bbox_inches='tight')
#plt.show()
