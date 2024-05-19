from SimpleAugment import SimpleAugSeq
import os
import time
import matplotlib.pyplot as plt
import gc
import sys
import math

if __name__ == '__main__':
    max_processes = os.cpu_count()
    if len(sys.argv) == 2:
        arg = sys.argv[1]
        if not arg.isdigit():
            exit(f'Please enter a number in the range 1 to {os.cpu_count()}.')
        max_processes = int(arg)
        if max_processes < 1:
            exit(f'Please enter a number in the range 1 to {os.cpu_count()}.')
        elif max_processes > os.cpu_count():
            exit(f'Please enter a number of processes less than or equal to {os.cpu_count()}.')
    path = '../test_data/raw/'
    save_path = '../test_data/aug/'
    copies = 2
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
        sa.deleteFiles(save_path)
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
    benchmark_dir = f'../benchmark_results/{copies}_copies'
    if not os.path.exists(benchmark_dir):
        os.makedirs(benchmark_dir)
    save_name = f'Copies{copies}_Processes{max_processes}_TimeVsProcesses'
    plt.savefig(os.path.join(benchmark_dir, f'{save_name}.png'))
    with open(os.path.join(benchmark_dir, f"{save_name}.txt"), "w") as f:
        for i in range(len(times)):
            f.write(f"Time to Augment: {times[i]} seconds with {processes[i]} processes\n")
        f.write(f"Minimum Time: {times[min]} seconds with {processes[min]} processes\n")
    plt.show()