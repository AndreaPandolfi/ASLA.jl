import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu, cg
import os

### IMPORTANT: before running this script, make sure to run 'python.jl' in Julia 

path_to_folder = os.path.join("paper", "real_data_example", "python")
chol_times = []; cg_times = []
dataset = 'GG' # 'GG' for G&G dataset and 'IE' for InstEval dataset

for i in range(1,7):
    # Load the CSV file into a pandas DataFrame
    try:
        spmatrix = pd.read_csv(os.path.join(path_to_folder, f'spmatrix_{dataset}{i}.csv')) # obtained with last formula
    except FileNotFoundError:
        continue

    ## COMPUTE Q AND th
    row = spmatrix['I'].values.astype(int) - 1 # -1 to convert to 0-based index
    col = spmatrix['J'].values.astype(int) - 1
    data = spmatrix['V'].values
    th = spmatrix['th'].dropna().values

    Q = coo_matrix((data, (row, col)))
    Q = Q.tocsr()

    Q_csc = Q.tocsc()  # Convert to CSC format for better performance in some cases

    ## BENCHMARKING
    # Ensure th is a column vector
    th = th.reshape(-1, 1)

    import timeit

    # Define the code to benchmark as functions
    def cholesky_solve():
        lu = splu(Q_csc)
        return lu.solve(th)

    def cg_solve():
        return cg(Q, th.flatten())

    # Run benchmarks
    num_trials = 10
    chol_time = timeit.timeit(cholesky_solve, number=num_trials) / num_trials
    # takes already so long with large matrix

    cg_time = timeit.timeit(cg_solve, number=num_trials) / num_trials

    chol_times.append(chol_time)
    cg_times.append(cg_time)

    print(f"Cholesky solution time: {chol_time:.6f} seconds")
    print(f"Conjugate Gradient solution time: {cg_time:.6f} seconds")

    # Save the results to a CSV file
    results_df = pd.DataFrame({
        'Cholesky Time (s)': chol_times,
        'CG Time (s)': cg_times
    })
    results_df.to_csv(os.path.join(path_to_folder, f'benchmark_python_{dataset}.csv'), index=False)

# cases = ["Random intercepts", "Nested effect", "Random slopes", "2 way interactions", "3 way interactions", "Everything"]
# results_df.index = cases
# with open(os.path.join(path_to_folder, f'benchmark_python_{dataset}.tex'), 'w') as f:
#     f.write(results_df.to_latex(float_format="%.3e"))
# results_df.to_csv(os.path.join(path_to_folder, f'benchmark_python_{dataset}.csv'))