from SimpleAugment import SimpleAugSeq
import os
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = os.path.join('..', 'test_data', 'raw')
    save_path = os.path.join('..', 'test_data', 'aug')
    copies = 4
    sass = [SimpleAugSeq(path=path, 
                                save_path=save_path, 
                                seed=1, 
                                num_copies=copies, 
                                names=[],
                                processes=i,
                                check=False) for i in range(1, os.cpu_count()+1)
                                ]
    times = [0 for i in range(1, os.cpu_count()+1)]
    print(f'This benchmark will test SimpleAugSeq with # processes in the range 1 to {os.cpu_count()} inclusive.')
    input('Press Enter to continue...')
    for i in range(len(sass)):
        sa = sass[i]
        sa.deleteFiles(save_path)
        start = time.time()
        sa.augment()
        end = time.time()
        times[i] = sa.duration
        
    min = times.index(min(times))
    processes = [i for i in range(1, os.cpu_count()+1)]
    plt.plot(processes, 
             times, 
             marker='o',
             linestyle='-', 
             color='b', 
             label='Data Points', zorder = 1)
    plt.scatter(processes[min], times[min], color='red', label='Minimum Time', zorder = 2)
    plt.text(processes[min], times[min] - 1, f'Minimum: {processes[min]} processes, {round(times[min],3)} seconds', 
         fontsize=10, ha='center', va='top', wrap=True)
    plt.title(f'Time to Augment (s) vs Number of Processes ({copies} copies per image)')
    plt.xlabel('Number of Processes')
    plt.ylabel('Time to Augment (s)')
    plt.xlim(0, os.cpu_count()+1)
    plt.ylim(0, max(times) + 10)
    plt.legend()
    benchmark_dir = os.path.join('..', 'benchmark_results')
    if not os.path.exists(benchmark_dir):
        os.makedirs(benchmark_dir)
    png = os.path.join(benchmark_dir, f'TimeVsProcesses_Copies{copies}.png')
    txt = os.path.join(benchmark_dir, f'TimeVsProcesses_Copies{copies}.txt')
    plt.savefig(png)
    with open(txt, "w") as f:
        for i in range(len(times)):
            f.write(f"Time to Augment: {times[i]} seconds with {processes[i]} processes\n")
        f.write(f"Minimum Time: {times[min]} seconds with {processes[min]} processes\n")
    plt.show()