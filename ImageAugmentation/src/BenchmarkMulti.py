from SimpleAugment import SimpleAugSeq
import os
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = '../test_data/raw/'
    save_path = '../test_data/aug/'
    copies = 4
    sass = [SimpleAugSeq(path=path, 
                                save_path=save_path, 
                                seed=1, 
                                num_copies=copies, 
                                names=[],
                                process=i,
                                check=False) for i in range(1, os.cpu_count()+1)
                                ]
    times = [0 for i in range(1, os.cpu_count()+1)]
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
    plt.ylim(0,20)
    plt.legend()
    plt.savefig(f'../TimeVsProcesses_Copies{copies}.png')
    plt.show()