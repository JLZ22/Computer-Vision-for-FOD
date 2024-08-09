from SimpleAugmenter import SimpleAugSeq
import os
import matplotlib.pyplot as plt
import gc
import sys
import math
from pathlib import Path
import Utils
import argparse

def parse_args():
    '''
    Parse command line arguments. The arguments are as follows:

    1. `--max-processes`: Max number of processes to use. 
        Default is the number of CPUs. This value cannot be less than 1.
    2. `--copies`: Number of copies per image. This value cannot be less than 1.
    '''
    parser = argparse.ArgumentParser(description='Benchmark SimpleAugSeq')
    parser.add_argument('--max-processes', type=int, default=os.cpu_count(), help='Max number of processes to use', metavar='INT')
    parser.add_argument('--copies', type=int, default=16, help='Number of copies per image', metavar='INT')
    return parser.parse_args()

def main():
    '''
    Perform a benchmark test on SimpleAugSeq with 1 to n processes inclusive. Save the 
    benchmark graph and results to the directory `../benchmark_results_simple_aug`. 
    The user can specify the max number of processes and the number of copies per image
    through the command line. Below is the usage of the command line arguments:

    ```
    usage: benchmark_simple_aug.py [-h] [--max-processes INT] [--copies INT]
    ```
    '''
    # Parse command line arguments
    args = parse_args()
    max_processes = min(args.max_processes, os.cpu_count())
    copies = args.copies
    if max_processes < 1:
        print('Max number of processes must be at least 1.')
        sys.exit(1)
    if copies < 1:
        print('Number of copies per image must be at least 1.')
        sys.exit(1)

    # temp directories to store copies of data
    out0 = Path('../test_data/out0')
    out1 = Path('../test_data/out1')
    Utils.delete_files(out0)
    Utils.delete_files(out1)
    
    # read directory
    pascalvoc_pairs = Path('../test_data/pascalvoc_pairs')
    Utils.pad_and_resize_square_in_directory(pascalvoc_pairs, out0)

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
    minimum = times.index(min(times))
    min_process = processes[minimum]
    min_time = times[minimum]

    # Plot the benchmark graph
    processes = [i for i in range(1, max_processes+1)]
    plt.plot(processes, 
             times, 
             marker='o',
             linestyle='-', 
             color='b', 
             label='Data Points', zorder = 1)
    plt.scatter(min_process, min_time, color='red', label='Minimum Time', zorder = 2)
    plt.text(min_process, min_time - math.ceil(max(times)/20), f'Minimum: {min_process} processes, {round(min_time,3)} seconds', 
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
        f.write(f"Minimum Time: {min_time} seconds with {min_process} processes\n")
    plt.show()

if __name__ == '__main__':
    main()