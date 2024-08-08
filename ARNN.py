import numpy as np
import random
from KarnaughMap import TruthTable, generate_random_truth_table, get_random_inputs_outputs, find_expression
import sys
from DataHandler import DataHandler
np.set_printoptions(suppress=True, precision=3, linewidth=sys.maxsize)

class ARNN:
    def __init__(self, n_inputs:int, n_hidden:int, n_outputs:int) -> None:
        self.n_inputs = n_inputs 
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.n_total = n_inputs + n_outputs + n_hidden

        self.A = np.zeros((self.n_total, self.n_total), dtype=int)
        self.W = np.zeros((self.n_total, self.n_total))
        self.V = np.zeros(self.n_total)
        self.E = np.zeros(self.n_total)
        
        self.input_nodes = np.arange(self.n_inputs)
        self.hidden_nodes = np.arange(self.n_inputs, self.n_inputs + self.n_hidden)
        self.output_nodes = np.arange(self.n_inputs + self.n_hidden, self.n_inputs + self.n_hidden + self.n_outputs)

        self._fully_connected()
        self.data_handler = DataHandler()

        self.aupdates = np.zeros(self.n_total, dtype=int)

    def _fully_connected(self):
        A = np.ones((self.n_total, self.n_total), dtype=int)
        np.fill_diagonal(A, 0)
        A[:, :self.n_inputs] = 0
        W = np.random.rand(self.n_total, self.n_total)*2 - 1
        self.A = A
        self.W = W
    
    # DATA HANDLING
    @property
    def data(self): return self.data_handler.data
    
    def store_data(self, save_params = True,**kwargs) -> None:
        if kwargs:      self.data_handler.add_data(**kwargs)
        if save_params: self.data_handler.add_data(A = self.A.copy(), W = self.W.copy())
    
    # OPTIMIZATION DECENTRALIZED
    def _forward_decentralized(self, node:int):
        self.V[node] = 1 / (1 + np.exp((-np.dot(self.V*self.A[:,node], self.W[:,node]))))

    def _error_gradient_decentralized(self, expected_outputs:np.array, node:int):
        if node in self.output_nodes: self.E[node] = (self.V[node]-expected_outputs[np.where(self.output_nodes == node)[0][0]])*0.15
        else: self.E[node] = np.dot(self.E*self.A[node], self.W[node])*(self.V[node]*(1-self.V[node]))
    
    def _backward_decentralized(self, node:int):
        self.W[:,node] = np.clip(self.W[:,node] - self.E[node]*self.V*self.A[:,node], -10, 10)

    def update_decentralized(self,  expected_outputs:np.array, 
                                    prune:bool, 
                                    update_parameters:bool, 
                                    weighted_diagonal:bool, 
                                    store_data:bool):
        
        nodes_to_update = np.arange(self.n_inputs, self.n_total)
        np.random.shuffle(nodes_to_update)
        
        for node in nodes_to_update:
            self._forward_decentralized(node)
            if update_parameters:
                self._error_gradient_decentralized(expected_outputs, node)
                self._backward_decentralized(node)
            if prune:
                self._eigen_pruning_decentralized(node, store_data=store_data, weighted_diagonal=weighted_diagonal)

            self.aupdates[node] += 1

    # PRUNING
    def _eigen_pruning_decentralized(self, node:int, store_data:bool, weighted_diagonal:bool):
        # Local Laplacian
        Al = np.zeros_like(self.A)
        Al[:,node] = self.A[:,node]
        Al[node] = self.A[node]

        if weighted_diagonal: Dl = np.diag(np.sum(Al*self.W, axis=0)) + np.diag(np.sum(Al*self.W, axis=1))
        else: Dl = np.diag(np.sum(Al, axis=1) + np.sum(Al, axis=0))        
        
        L = Dl - Al*self.W
        
        # Eigenvalues
        eigen, eigenvectors = np.linalg.eig(L)
        im_eigen = np.imag(eigen)

        if store_data: self.store_data(N = node, R = np.real(eigen), I = im_eigen, save_params=False)
        #print(f'Node: {node} Eigen im,: {im_eigen}')
        aupdate = len(self.data['N']) if 'N' in self.data.keys() else 0

        if np.all(im_eigen == 0): return
        max_index_in = np.argmax(im_eigen)
        max_value_in = im_eigen[max_index_in]

        if -max_value_in in im_eigen and max_value_in != 0:
            min_index_in = np.argmin(im_eigen)
            from_node = min_index_in
            to_node = max_index_in
            if self.A[from_node][to_node] != 0:
                print(f'    Removing edge from {from_node} to {to_node}')
                self.A[from_node][to_node] = 0
                self.W[from_node][to_node] = 0
                if store_data: self.store_data(aupdate = self.aupdates[node], Np = node, edge_removed = [from_node, to_node], save_params=False)

if __name__ == '__main__':
    pass