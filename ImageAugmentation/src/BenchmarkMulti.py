from SimpleAugment import SimpleAugSeq
import os
import time
import matplotlib.pyplot as plt
import gc
import sys
import math
from pathlib import Path
import utils

def check_first_arg(arg):
    if not arg.isdigit():
        exit(f'Please enter a integer in the range 1 to {os.cpu_count()} for the first argument.')
    max_processes = int(arg)
    if max_processes < 1:
        exit(f'Please enter a integer in the range 1 to {os.cpu_count()} for the first argument.')
    elif max_processes > os.cpu_count():
        exit(f'Please enter a integer of processes less than or equal to {os.cpu_count()} for the first argument.')
    return max_processes

def check_second_arg(arg):
    if not arg.isdigit() or int(arg) <= 0:
        exit(f'Please enter a positive integer for the second argument.')
    return int(arg)

if __name__ == '__main__':
    max_processes = os.cpu_count()
    copies = 8
    if len(sys.argv) > 3:
        exit('Too many arguments.')
    if len(sys.argv) >= 2:
        max_processes = check_first_arg(sys.argv[1])
    if len(sys.argv) == 3:
        copies = check_second_arg(sys.argv[2])
    path = Path('..', 'test_data', 'raw_pairs')
    save_path = Path('..', 'test_data', 'aug_pairs')
    sass = [SimpleAugSeq(path=path, 
                                save_path=save_path, 
                                seed=1, 
                                num_copies=copies, 
                                names=[],
                                processes=i,
                                check=False) for i in range(1, max_processes+1)
                                ]
    times = [0 for i in range(1, max_processes+1)]
    print(f'This benchmark will test SimpleAugSeq with # processes in the range 1 to {max_processes} inclusive.')
    print(f'The number of copies per image is {copies}.')
    input('Press Enter to continue...')
    for i in range(len(sass)):
        sa = sass[i]
        utils.deleteFiles(save_path)
        start = time.time()
        sa.augment()
        end = time.time()
        times[i] = sa.duration
        del sa
        gc.collect()
        
    min = times.index(min(times))
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
    benchmark_dir = Path('..', f'benchmark_results/{copies}_copies')
    if not benchmark_dir.exists() or not benchmark_dir.is_dir():
        os.makedirs(benchmark_dir)
    save_name = f'Copies{copies}_Processes{max_processes}_TimeVsProcesses'
    png = Path(benchmark_dir, f'{save_name}.png')
    txt = Path(benchmark_dir, f'{save_name}.txt')
    plt.savefig(png)
    with txt.open(mode='w') as f:
        for i in range(len(times)):
            f.write(f"Time to Augment: {times[i]} seconds with {processes[i]} processes\n")
        f.write(f"Minimum Time: {times[min]} seconds with {processes[min]} processes\n")
    plt.show()