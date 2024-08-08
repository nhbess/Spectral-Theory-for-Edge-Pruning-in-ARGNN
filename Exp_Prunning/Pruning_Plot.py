import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from _colors import create_palette, show_palette

EXP_FOLDER = os.path.dirname(os.path.abspath(__file__))
EXP_NAME = 'Pruning'
filename = f'{EXP_NAME}_Results.json'
with open(os.path.join(EXP_FOLDER, filename), 'r') as f:
    results = json.load(f)

for key in results:
    print(f'{key}, {np.mean(results[key]["edges_before"])}, {np.mean(results[key]["edges_after"])}, {np.mean(results[key]["mean_errors"])}')

gates_names = ['AND','OR','XOR']

palette = create_palette(20)
show_palette(palette, save=True, name='palette')

color_NONE = palette[10]
color_D = palette[5]
color_DW = palette[14]

color_D_percentual = palette[2]
color_DW_percentual = palette[17]


fig, axs = plt.subplots(len(gates_names), 2, figsize=(5, 1.5*len(gates_names)), sharex=True)

for i, gate in enumerate(gates_names):
    
    results_gate = {k:v for k,v in results.items() if gate in k}
    H = [int(k.split('_')[-1]) for k in results_gate.keys()]
    H = sorted(list(set(H)))

    def get_errors(gate, prunning_mode):
        results = {k:v for k,v in results_gate.items() if prunning_mode in k}        
        for h in H:
            key = f'{gate}_{prunning_mode}_{h}'
            errors = results[key]['mean_errors']
            yield np.mean(errors), np.std(errors)
        
    prune_None_mean, prune_None_std = zip(*get_errors(gate, 'None'))
    prune_D_mean, prune_D_std = zip(*get_errors(gate, 'D'))
    prune_DW_mean, prune_DW_std = zip(*get_errors(gate, 'Dw'))
    
    axs[i,0].errorbar(H, prune_None_mean, yerr=prune_None_std, label='$None$', color=color_NONE, marker='o', alpha=0.9, linestyle=':')
    axs[i,0].errorbar(H, prune_D_mean, yerr=prune_D_std, label='$D$', color=color_D, marker='o', alpha=0.9)
    axs[i,0].errorbar(H, prune_DW_mean, yerr=prune_DW_std, label='$D_{W}$', color=color_DW, marker='o', alpha=0.9, linestyle='--')
    
    def get_edges(gate, prunning_mode):
        results = {k:v for k,v in results_gate.items() if prunning_mode in k}
        for h in H:
            key = f'{gate}_{prunning_mode}_{h}'
            edges_before = np.array(results[key]['edges_before'])
            edges_after = np.array(results[key]['edges_after'])
            pruned_edges = edges_before - edges_after
            pruned_edges_percentual = pruned_edges/edges_before * 100
            yield np.mean(pruned_edges), np.std(pruned_edges), np.mean(pruned_edges_percentual), np.std(pruned_edges_percentual)

    edges_D_mean, edges_D_std, edges_D_mean_percentual, edges_D_std_percentual = zip(*get_edges(gate, 'D'))
    edges_DW_mean, edges_DW_std, edges_DW_mean_percentual, edges_DW_std_percentual = zip(*get_edges(gate, 'Dw'))
    

    ax2 = axs[i, 1].twinx()
    ax2.errorbar(H, edges_D_mean_percentual, yerr=edges_D_std_percentual, label='$D$%', color=color_D_percentual, marker='o', alpha = 0.7, linestyle='--', zorder = 1)
    ax2.errorbar(H, edges_DW_mean_percentual, yerr=edges_DW_std_percentual, label='$D_{W}$%', color=color_DW_percentual, marker='o', alpha = 0.7, linestyle='--', zorder = 1)
    ax2.set_ylabel('Percentual')  # Set label for the right y-axis

    axs[i,1].errorbar(H, edges_D_mean, yerr=edges_D_std, label='$D$', color=color_D, marker='o', alpha = 1, zorder = 3)
    axs[i,1].errorbar(H, edges_DW_mean, yerr=edges_DW_std, label='$D_{W}$', color=color_DW, marker='o', alpha = 1, zorder = 3)


for i, gn in enumerate(gates_names):
    #axs[i,0].set_title(gn)
    axs[i,0].set_ylabel(f'{gn}')
    axs[i,1].set_ylabel(f'Absolute')
    axs[i,0].set_xticks([]) # remove ticks
    axs[i,1].set_xticks([]) # remove ticks
    #axs[i,-1].set_xlabel('Hidden Nodes')

#axs[0,0].set_ylabel('Error')
#axs[0,1].set_ylabel('Edges Removed')
axs[0,0].set_title('Error')
axs[0,1].set_title('Edges Removed')

axs[-1,0].set_xlabel('Hidden Nodes')
#add ticks
axs[-1,0].set_xticks(H)
axs[-1,1].set_xlabel('Hidden Nodes')
axs[-1,1].set_xticks(H)

axs[0,0].legend(fontsize='small')
# Gather handles and labels from both axes
lines_1, labels_1 = axs[i, 1].get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
# Combine handles and labels
lines = lines_1 + lines_2
labels = labels_1 + labels_2
# Create a single legend
axs[-1, 1].legend(lines, labels, framealpha=0.5, fontsize='small', loc='lower right', ncol=2)


plt.subplots_adjust(hspace=0.5)

plt.tight_layout()


filename = 'Pruning_Results.png'
plt.savefig(os.path.join(EXP_FOLDER,filename), dpi=600, bbox_inches='tight')
#plt.show()
