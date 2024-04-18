import argparse
import sys
import os
import joblib
import numpy as np
from scipy.optimize import minimize
from npy_append_array import NpyAppendArray


#
# Target functions
#

def Himmelblau(x):
    dim = len(x)
    func = 0
    for i in range(dim-1):
        func += (x[i]**2 + x[i+1] - 11)**2 + (x[i] + x[i+1]**2 -7)**2
    func += 1
    func = np.log(func)

    if dim == 2:
        pass
    elif dim == 3:
        func -= 0.265331837897597
    elif dim == 4:
        func -= 1.7010318616354436
    elif dim == 5:
        func -= 2.3001107745553155
    elif dim == 6:
        func -= 2.8576426513378994
    else:
        raise Exception("We don't know the minimum value for Himmelblau in this number of dimensions.")

    return func


def Rosenbrock(x):
    dim = len(x)
    func = 0
    for i in range(dim-1):
        func += 100*(x[i+1]-x[i]**2)**2 + (1 - x[i])**2
    
    return func


# Register target functions in this dictionary
target_functions = {
    'himmelblau': Himmelblau,
    'rosenbrock': Rosenbrock,
}



#
# Main script
#

# Read command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('target', type=str, help=f"Name of the target function. Registered functions are: {', '.join(target_functions.keys())}")
parser.add_argument('binning', nargs='+', type=float, help="A list on the form: x1_min x1_max n_bins_x1  x2_min x2_max n_bins_x2  ...")
parser.add_argument('--tol', type=float, help="Tolerance parameter forwarded to scipy.minimize")
parser.add_argument('--method', type=str, default='L-BFGS-B', help="A method name forwarded to scipy.minimize, e.g. L-BFGS-B")
parser.add_argument('--nprocs', type=int, default=1, help="Number of parallel processes")
parser.add_argument('--buffer', type=int, default=int(1e4), help="Number of results stored in memory before writing to file")
parser.add_argument('--outname', type=str, default='', help="The base name for all output files")
parser.add_argument('--saveindices', dest='saveindices', default=False, action='store_true', help="If used, the set of all bin indices will be written to file")
parser.add_argument('--verbose', dest='verbose', default=False, action='store_true', help="If used, the result of every optimization task is printed to screen (can lead to significantly longer runtime)")
args = parser.parse_args()

# Save arguments in dedicated variables (for no other reason than that I prefer it)
target_input = args.target.lower()
binning_input = args.binning
tol_input = args.tol
nprocs_input = args.nprocs
method_input = args.method
buffer_input = args.buffer
outname_input = args.outname
verbose_input = args.verbose
save_indices_input = args.saveindices

if outname_input != '':
    outname_input = outname_input + '_'

# Consistency checks
if target_input not in target_functions.keys():
    print(f"error: unknown target function '{target_input}'. The registered target functions are: {', '.join(target_functions.keys())}")
    sys.exit()

if len(binning_input) % 3 != 0:
    print("error: the length of the 'binning' input list must be a multiple of three (x1_min x1_max n_bins_x1  x2_min x2_max n_bins_x2  ...)")
    sys.exit()

# Set the target function
target_func = target_functions[target_input]

# Create a list of tuples: [(x1_min, x1_max, n_bins_x1), (x2_min, x2_max, n_bins_x2), ...]
binning_tuples = []
for i in range(0, len(binning_input), 3):
    binning_tuples.append( (binning_input[i], binning_input[i+1], int(binning_input[i+2]) ) )

# Get number of dimensions
n_dims = len(binning_tuples)

# Get number of bins per dimension
n_bins_per_dim = [bt[2] for bt in binning_tuples]

# Compute total number of tasks as n_bins_x1 * n_bins_x2 * ...
n_tasks = np.prod(n_bins_per_dim)
print(f"We have {n_tasks} tasks to do...")

# No need to use more processes than there are tasks do to
nprocs = min(n_tasks, nprocs_input)

# List of per-process output file names
proc_result_files = [f"{outname_input}result_proc_{p}.npy" for p in range(nprocs)]
proc_indices_files = [f"{outname_input}indices_proc_{p}.npy" for p in range(nprocs)]


# Create a list with the number of tasks per process
tasks_per_process = int(n_tasks / nprocs)
n_tasks_per_process = [tasks_per_process]*nprocs
n_remaining_tasks = n_tasks - tasks_per_process * nprocs 
p = 0
while n_remaining_tasks > 0:
    n_tasks_per_process[p] += 1
    n_remaining_tasks -= 1
    p += 1
    p = p % nprocs


# A helper function that returns bounds and starting point for
# task number i on process p
def get_task_tuple(p,i):

    # Compute the bin index for each dimension
    bin_indices = [0]*n_dims

    n = 0
    for j in range(p):
        n += n_tasks_per_process[j]
    n += i

    for d in range(n_dims):
        sd = int(np.prod(n_bins_per_dim[d+1:]))
        nd = n_bins_per_dim[d]
        d_index = (n // sd) % nd
        bin_indices[d] = d_index

    bin_indices = tuple(bin_indices)

    # Use the bin indices to compute the bounds and starting point
    task_bounds = []
    task_start_point = np.zeros(n_dims)

    for d in range(n_dims):

        bin_index_d = bin_indices[d]

        xmin, xmax, nbins = binning_tuples[d]
        bin_width = (xmax - xmin) / (nbins)

        bin_xmin = xmin + bin_index_d * bin_width
        bin_xmax = bin_xmin + bin_width
        task_bounds.append((bin_xmin, bin_xmax))

        task_start_point[d] =  xmin + (bin_index_d + 0.5) * bin_width 

    return task_bounds, task_start_point, bin_indices


# The function that will be called by joblib.Parallel
def do_task_batch(p):

    # Open indices file for appending
    proc_indices_file = proc_indices_files[p]
    with NpyAppendArray(proc_indices_file, delete_if_exists=True) as indices_f:

        my_indices = None
        if save_indices_input:
            my_indices = np.zeros((buffer_input,n_dims), dtype=int)

        # Open result file for appending
        proc_result_file = proc_result_files[p]
        with NpyAppendArray(proc_result_file, delete_if_exists=True) as result_f:

            # my_task_indices = task_indices_per_process[p]
            n_tasks_in_batch = n_tasks_per_process[p]

            my_results = np.zeros(buffer_input)

            # Do all the tasks in this batch
            buffer_i = 0
            for task_i in range(n_tasks_in_batch):

                bounds, start_point, bin_indices = get_task_tuple(p, task_i)
                if save_indices_input:
                    my_indices[buffer_i] = np.array(bin_indices)

                # Do the optimization and store the result
                res = minimize(target_func, start_point, bounds=bounds, method=method_input, tol=tol_input)
                my_results[buffer_i] = res.fun

                buffer_i += 1

                # Print every result? Slow!
                if verbose_input:
                    bounds_str = ", ".join([f"({b[0]: .4e},{b[1]: .4e})" for b in bounds])
                    start_point_str = "(" + ",".join([f"{xj: .4e}" for xj in start_point]) + ")"
                    print(f"Task (batch,index): ({p},{task_i})   Bounds: {bounds_str}   Starting point: {start_point_str}   Result: {res.fun: .10e}")

                # Print progress indicator
                if (task_i+1) % 1000 == 0:
                    print(f"Process {p}: done with task {task_i+1} of {n_tasks_in_batch}")

                # Save result to file when the buffer is full
                if (buffer_i) % buffer_input == 0:
                    print(f"Process {p}: Appending results to file {proc_result_file}")

                    result_f.append(my_results)
                    my_results = np.zeros(buffer_input)

                    if save_indices_input:
                        indices_f.append(my_indices)
                        my_indices = np.zeros((buffer_input,n_dims), dtype=int)

                    buffer_i = 0

            # Save the remaining results from the buffer
            if buffer_i > 0:
                result_f.append(my_results[:buffer_i])
                if save_indices_input:
                    indices_f.append(my_indices[:buffer_i])


# Run tasks in parallel
joblib.Parallel(n_jobs=nprocs)(joblib.delayed(do_task_batch)(p) for p in range(nprocs))


# Combine result files into a single file
result_file = f"{outname_input}result.npy"
print(f"Writing array with combined results to file {result_file}")
with NpyAppendArray(result_file, delete_if_exists=True) as result_f:
    for p in range(nprocs):
        proc_data = np.load(proc_result_files[p])
        result_f.append(proc_data)
# Now remove the process files
for f in proc_result_files:
    try:
        os.remove(f)
    except OSError:
        pass

# Combine indices files into a single file
if save_indices_input:
    indices_file = f"{outname_input}indices.npy"
    print(f"Writing array with combined set of bin indices to file {indices_file}")
    with NpyAppendArray(indices_file, delete_if_exists=True) as indices_f:
        for p in range(nprocs):
            proc_indices = np.load(proc_indices_files[p])
            indices_f.append(proc_indices)
# Now remove the process files
for f in proc_indices_files:
    try:
        os.remove(f)
    except OSError:
        pass

# Write an output file with the bin limits in each dimension
bin_limits_file = f"{outname_input}bin_limits.npy"
print(f"Writing the bin limits for each dimension to file {bin_limits_file}")
with NpyAppendArray(bin_limits_file, delete_if_exists=True) as bin_limits_f:
    bin_limits = np.array([ np.linspace(bt[0], bt[1], bt[2]+1) for bt in binning_tuples ])
    bin_limits_f.append(bin_limits)

print(f"Done")

