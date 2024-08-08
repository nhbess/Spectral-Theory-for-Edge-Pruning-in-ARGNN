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
EXP_NAME = 'Timing'

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
    N_ITERATIONS = 1000
    N_RUNS = 5
    H = [2,5,10]

    results = {}
    for tt, tt_name in zip(tts, tts_names):
        for prunning_mode in ['D', 'Dw']:
            if prunning_mode == 'D': weighted_diagonal = False
            else: weighted_diagonal = True

            for h in H:
                run_edges = []
                run_errors = []

                for run in range(N_RUNS):
                    print(f'{tt_name} H: {h} Run: {run} Prune: {prunning_mode}')
                    inputs,outputs = get_random_inputs_outputs(truth_table=tt, n_inputs=N_INPUTS, size=N_ITERATIONS, random_gen=None)                    
                    m = ARNN(n_inputs=N_INPUTS, n_hidden=h, n_outputs=N_OUTPUTS)   
                    errors, edges = train(m, inputs, outputs, N_AUPDATES, 
                          prune=True, 
                          weighted_diagonal=weighted_diagonal,
                          store_data=False,
                          return_training_errors=True,
                          return_edge_number=True)
                    
                    run_edges.append([int(e) for e in edges])
                    run_errors.append(errors)

                key = f"{tt_name}_{prunning_mode}_{h}"
                results[key] = {
                "edges": run_edges,
                "mean_errors": run_errors
                }

    filename = f'{EXP_NAME}_Results.json'
    with open(os.path.join(EXP_FOLDER, filename), 'w') as f:
        json.dump(results, f)
