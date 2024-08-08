import numpy as np
from itertools import product

# Truth Table for AND, OR, XOR
class TruthTable:
    tt_AND = np.array([[0,0,0], 
                       [0,1,0], 
                       [1,0,0], 
                       [1,1,1]])

    tt_OR  = np.array([[0,0,0],
                       [0,1,1],
                       [1,0,1],
                       [1,1,1]])

    tt_XOR = np.array([[0,0,0],
                       [0,1,1],
                       [1,0,1],
                       [1,1,0]])


# Generate a random truth table with n_inputs and n_outputs
def generate_random_truth_table(n_inputs:int, n_outputs:int) -> np.ndarray:
    n_rows = 2 ** n_inputs
    inputs = np.array(list(product([0, 1], repeat=n_inputs)))
    outputs = np.random.randint(2, size=(n_rows, n_outputs))
    truth_table = np.hstack((inputs, outputs))
    return truth_table

# Get random inputs and outputs from a truth table
def get_random_inputs_outputs(truth_table: np.ndarray, n_inputs: int, size:int, random_gen:np.random.Generator=None) -> tuple:
    n_rows = truth_table.shape[0]
    if random_gen is not None: 
        indices = random_gen.integers(n_rows, size=size)
    else:
        indices = np.random.randint(n_rows, size=size)
    inputs = truth_table[indices, :n_inputs]
    outputs = truth_table[indices, n_inputs:]
    return inputs, outputs

# Getting the expression from a truth table
def _truth_table_to_minterms(truth_table, n_vars, output_index):
    minterms = []
    for row in truth_table:
        if row[n_vars + output_index] == 1:
            minterms.append(tuple(row[:n_vars]))
    return minterms

def _combine_terms(term1, term2):
    combined = []
    differences = 0
    for b1, b2 in zip(term1, term2):
        if b1 == b2:
            combined.append(b1)
        else:
            combined.append(-1)  # Use -1 to denote a don't care condition
            differences += 1
    if differences == 1:
        return tuple(combined)
    return None

def _quine_mccluskey(minterms, n_vars):
    def group_minterms(minterms):
        grouped = [[] for _ in range(n_vars + 1)]
        for minterm in minterms:
            count = sum(minterm)
            grouped[count].append(minterm)
        return grouped
    
    def _find_prime_implicants(grouped):
        prime_implicants = set()
        while True:
            new_grouped = [[] for _ in range(len(grouped) - 1)]
            combined = set()
            for i in range(len(grouped) - 1):
                for term1 in grouped[i]:
                    for term2 in grouped[i + 1]:
                        combined_term = _combine_terms(term1, term2)
                        if combined_term:
                            new_grouped[sum(1 for bit in combined_term if bit == 1)].append(combined_term)
                            combined.add(term1)
                            combined.add(term2)
            prime_implicants.update(set(sum(grouped, [])) - combined)
            grouped = new_grouped
            if not any(grouped):
                break
        return prime_implicants
    
    minterms = [tuple(map(int, m)) for m in minterms]
    grouped = group_minterms(minterms)
    prime_implicants = _find_prime_implicants(grouped)
    return prime_implicants

def _prime_implicants_to_expression(prime_implicants, n_vars):
    terms = []
    for implicant in prime_implicants:
        term = []
        for i, bit in enumerate(implicant):
            if bit == 1:
                term.append(f'x{i+1}')
            elif bit == 0:
                term.append(f"x{i+1}'")
        terms.append(''.join(term))
    return ' + '.join(terms)

def find_expression(tt:np.array, n_inputs:int, n_outputs:int):
    
    for output_index in range(n_outputs):
        minterms = _truth_table_to_minterms(tt, n_inputs, output_index)
        print(f"\nOutput {output_index + 1}: Minterms = {minterms}")
        if not minterms:
            expression = '0'  # No minterms mean the output is always 0
            prime_implicants = set()
        else:
            prime_implicants = _quine_mccluskey(minterms, n_inputs)
            if not prime_implicants:
                expression = '1'  # If prime implicants are empty, the function covers all cases
            else:
                expression = _prime_implicants_to_expression(prime_implicants, n_inputs)
        print(f"Prime Implicants: {prime_implicants}")
        print(f"Simplified Expression for Output {output_index + 1}: {expression}")

if __name__ == '__main__':
    n_inputs = 2
    n_outputs = 1

    truth_table = generate_random_truth_table(n_inputs, n_outputs)
    print(truth_table)
    find_expression(truth_table, n_inputs, n_outputs)