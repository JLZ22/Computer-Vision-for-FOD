import multiprocessing
#This pogram will just spike you're CPU usage
#It is not usefull

def worker(name):
    #worker function
    #print ('Worker')
    x = 0
    while x < 1000:
        x += 1
    return

if __name__ == '__main__':
    jobs = []
    for i in range(50):
        p = multiprocessing.Process(target=worker(str(i)))
        jobs.append(p)
        p.start()