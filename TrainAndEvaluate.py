from ARNN import ARNN
import numpy as np
import sys
from tqdm import tqdm

def train(m:ARNN, 
          inputs:np.array, 
          outputs:np.array, 
          n_aupdates:int, 
          prune:bool, 
          weighted_diagonal:bool,
          store_data:bool,
          return_training_errors:bool,
          return_edge_number:bool):
    '''
    Train the model m using the inputs and outputs, the model is updated n_aupdates times for each input
    '''
    if return_training_errors: errors = []
    if return_edge_number: edges = [m.A.sum()]
    for input , expected_output in zip(inputs,outputs):
        m.V[m.input_nodes] = input
        for i in range(n_aupdates):
            m.update_decentralized(expected_output, prune=prune, update_parameters=True, weighted_diagonal=weighted_diagonal, store_data=store_data)
    
        if store_data:
            error = np.mean(np.abs(m.V[m.output_nodes] - expected_output))
            m.store_data(error=error, save_params=False)
    
        if return_training_errors:
            error = np.mean(np.abs(m.V[m.output_nodes] - expected_output))
            errors.append(error)
        
        if return_edge_number:
            edges.append(m.A.sum())
                    
    if return_edge_number and return_training_errors: return errors, edges    
    if return_edge_number: return edges
    if return_training_errors: return errors
    
def evaluate_performance_thruth_table(m:ARNN, truth_table:np.array, n_aupdates:int):
    '''
    Evaluate the performance of the model m using the truth table, 
    the inputs are shuffled and the model is updated for update_steps.
    It returns the errors and the mean error of the output nodes after n update_steps (only the last one).
    errors: mean error for each input in the truth table (individual if only one input)
    '''
    indexes = np.arange(len(truth_table))
    np.random.shuffle(indexes)
    truth_table = truth_table[indexes]

    X = truth_table[:, :m.n_inputs]
    Y = truth_table[:, m.n_inputs:]

    errors = []
    for input,expected_output in zip(X,Y):
        m.V[m.input_nodes] = input
        for _ in range(n_aupdates):
            m.update_decentralized(expected_output, 
                                   prune=False, 
                                   update_parameters=False, 
                                   weighted_diagonal=False, 
                                   store_data=False,)
        error = np.mean(np.abs(m.V[m.output_nodes] - expected_output))
        errors.append(error)

    return errors, np.mean(errors)

def evaluate_performance_MNIST(m:ARNN, X:np.array,Y:np.array, n_aupdates:int):
    good = 0
    bad = 0
    for input,expected_output in zip(X,Y):
        m.V[m.input_nodes] = input
        for _ in range(n_aupdates):
            m.update(expected_output, prune=False, update_parameters=False)     
        output_values = m.V[m.output_nodes]
        if np.argmax(output_values) == np.argmax(expected_output):
            good += 1
        else:
            bad += 1
    return good/(good+bad)