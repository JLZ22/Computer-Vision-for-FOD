from SimpleAugmenter import SimpleAugSeq
import os
import matplotlib.pyplot as plt
import gc
import sys
import math
from pathlib import Path
import Utils

def check_first_arg(arg: str):
    '''
    Check if the first argument is a positive integer in the range 1 to the number of cpus.
    If the argument is not a positive integer, the program will exit. 

    arg: The first argument from the command line.
    '''
    if not arg.isdigit():
        exit(f'Please enter a integer in the range 1 to {os.cpu_count()} for the first argument.')
    max_processes = int(arg)
    if max_processes < 1:
        exit(f'Please enter a integer in the range 1 to {os.cpu_count()} for the first argument.')
    elif max_processes > os.cpu_count():
        exit(f'Please enter a integer of processes less than or equal to {os.cpu_count()} for the first argument.')
    return max_processes

def check_second_arg(arg: str):
    '''
    Check if the second argument is a positive integer. If the argument is not a positive integer, 
    the program will exit.

    arg: The second argument from the command line.
    '''
    if not arg.isdigit() or int(arg) <= 0:
        exit(f'Please enter a positive integer for the second argument.')
    return int(arg)

def main():
    '''
    Perform a benchmark test on SimpleAugSeq with 1 to n processes inclusive. Save the 
    benchmark graph and results to the directory `../benchmark_results_simple_aug`. 
    The user can specify the max number of processes.
    '''

    # temp directories to store copies of data
    out0 = Path('../test_data/out0')
    out1 = Path('../test_data/out1')
    Utils.delete_files(out0)
    Utils.delete_files(out1)
    
    # read directory
    pascalvoc_pairs = Path('../test_data/pascalvoc_pairs')
    Utils.pad_and_resize_square_in_directory(pascalvoc_pairs, out0)

    # Initialize variables
    max_processes = os.cpu_count()
    max_processes = 16
    copies = 64

    # Check command line arguments
    if len(sys.argv) > 3:
        exit('Too many arguments.')
    if len(sys.argv) >= 2:
        max_processes = check_first_arg(sys.argv[1])
    if len(sys.argv) == 3:
        copies = check_second_arg(sys.argv[2])

    # Create array of augmenters with different number of processes
    sass = [SimpleAugSeq(read_path=out0, 
                                save_path=out1, 
                                seed=1, 
                                num_copies=copies, 
                                names=[],
                                processes=i,
                                check=False) for i in range(1, max_processes+1)
                                ]
    
    # Initialize array representing the time to augment for each number of processes
    times = [0 for _ in range(1, max_processes+1)]

    print(f'This benchmark will test SimpleAugSeq with 1 to {max_processes} processes inclusive.')
    print(f'The number of copies per image is {copies}.')
    input('Press Enter to continue...')

    # Perform benchmark test
    for i in range(len(sass)):
        sa = sass[i]
        Utils.delete_files(out1)
        sa.augment()
        times[i] = sa.duration
        del sa
        gc.collect()
        
    # Find the minimum time and plot the benchmark graph
    min = times.index(min(times))

    # Plot the benchmark graph
    processes = [i for i in range(1, max_processes+1)]
    plt.plot(processes, 
             times, 
             marker='o',
             linestyle='-', 
             color='b', 
             label='Data Points', zorder = 1)
    plt.scatter(processes[min], times[min], color='red', label='Minimum Time', zorder = 2)
    plt.text(processes[min], times[min] - math.ceil(max(times)/20), f'Minimum: {processes[min]} processes, {round(times[min],3)} seconds', 
         fontsize=10, ha='center', va='top', wrap=True)
    plt.title(f'Time to Augment (s) vs Number of Processes ({copies} copies per image)')
    plt.xlabel('Number of Processes')
    plt.ylabel('Time to Augment (s)')
    plt.xlim(0, max_processes+1)
    plt.ylim(0, max(times) + 10)
    plt.legend()

    # Save the benchmark graph
    benchmark_dir = Path('..', f'benchmark_results_simple_aug/{copies}_copies')
    if not benchmark_dir.exists() or not benchmark_dir.is_dir():
        os.makedirs(benchmark_dir)
    save_name = f'Copies{copies}_Processes{max_processes}_TimeVsProcesses'
    png = Path(benchmark_dir, f'{save_name}.png')
    txt = Path(benchmark_dir, f'{save_name}.txt')
    plt.savefig(png)

    # Save the benchmark results in text file
    with txt.open(mode='w') as f:
        for i in range(len(times)):
            f.write(f"Time to Augment: {times[i]} seconds with {processes[i]} processes\n")
        f.write(f"Minimum Time: {times[min]} seconds with {processes[min]} processes\n")
    plt.show()

if __name__ == '__main__':
    main()