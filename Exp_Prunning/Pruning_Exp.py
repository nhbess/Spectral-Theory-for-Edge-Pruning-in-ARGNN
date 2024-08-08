#add mother folder to path
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random

import numpy as np

from ARNN import ARNN
from KarnaughMap import TruthTable, get_random_inputs_outputs
from TrainAndEvaluate import evaluate_performance_thruth_table, train


EXP_FOLDER = os.path.dirname(os.path.abspath(__file__))
EXP_NAME = 'Pruning'

if __name__ == '__main__':

    seed = np.random.randint(0, 1000000)
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    print(f'Seed: {seed}')

    tts = [TruthTable.tt_AND, TruthTable.tt_OR, TruthTable.tt_XOR]
    tts_names = ['AND', 'OR', 'XOR']
    
    N_INPUTS = 2
    N_OUTPUTS = 1

    N_AUPDATES = 10
    N_ITERATIONS = 2000
    N_RUNS = 10
    H = np.arange(0, 6, 1).tolist()

    results = {}
    for tt, tt_name in zip(tts, tts_names):
        for prunning_mode in ['D', 'Dw', 'None']:
            if prunning_mode == 'None':
                prune = False
                weighted_diagonal = False
            else:
                prune = True
                if prunning_mode == 'D': weighted_diagonal = False
                else: weighted_diagonal = True

            for h in H:
                run_edges_before = []
                run_edges_after = []
                run_mean_errors = []

                for run in range(N_RUNS):
                    print(f'{tt_name} H: {h} Run: {run} Prune: {prunning_mode}')
                    inputs,outputs = get_random_inputs_outputs(truth_table=tt, n_inputs=N_INPUTS, size=N_ITERATIONS, random_gen=None)                    
                    m = ARNN(n_inputs=N_INPUTS, n_hidden=h, n_outputs=N_OUTPUTS)   
                    edges_before = np.sum(m.A)
                    train(m, inputs, outputs, N_AUPDATES, 
                          prune=prune, 
                          weighted_diagonal=weighted_diagonal,
                          store_data=False,
                          return_training_errors=False,)
                    edges_after = np.sum(m.A)
                    errors, mean_error = evaluate_performance_thruth_table(m, tt, N_AUPDATES)

                    run_edges_before.append(int(edges_before))
                    run_edges_after.append(int(edges_after))
                    run_mean_errors.append(mean_error)
                
                key = f"{tt_name}_{prunning_mode}_{h}"
                results[key] = {
                "edges_before": run_edges_before,
                "edges_after": run_edges_after,
                "mean_errors": run_mean_errors
                }
                print(f'{key}, {np.mean(run_edges_before)}, {np.mean(run_edges_after)}, {np.mean(run_mean_errors)}')

    filename = f'{EXP_NAME}_Results.json'
    with open(os.path.join(EXP_FOLDER, filename), 'w') as f:
        json.dump(results, f)
