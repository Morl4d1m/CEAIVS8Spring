from multiprocessing import Pool
import os, math, random, time, statistics

def piSerial(numSamples):
    inCircle = 0
    for _ in range(numSamples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inCircle += 1
    return inCircle#4*inCircle/numSamples

def piParallel(numSamples,processes=4):
    samplesPerProcess = numSamples//processes
    jobs = [samplesPerProcess]*processes
    with Pool(processes=processes) as pool:
        results = pool.map(piSerial, jobs)
    return 4*sum(results)/numSamples

def granularityTest(totalWork, chunkSize, nProcessors):
    nChunks = totalWork//chunkSize
    tasks = [chunkSize]*nChunks
    t1 = time.perf_counter()
    if nProcessors == 1:
        results =[piSerial(i) for i in tasks]
    else:
        with Pool(processes=nProcessors) as pool:
            results = pool.map(piSerial, tasks)
    return time.perf_counter()-t1, 4*sum(results)/totalWork

if __name__ == '__main__':
    numSamples = 10000000
    for processors in range(1,os.cpu_count()+1):
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            piEstimateSerial = piSerial(numSamples)
            piEstimateParallel = piParallel(numSamples)
            times.append(time.perf_counter() - t0)
        tSerial = statistics.median(times)
        print(f"{processors:2d} workers: {tSerial:.3f}s pi={piEstimateParallel:.6f}")
    print(f"pi estimate: {piEstimateSerial:.6f} (error: {abs(piEstimateSerial-math.pi):.6f})")
    print(f"Serial time: {tSerial:.3f}s")

    nProcessors = os.cpu_count()//2
    chunkSizes = [10,100,1000,10000,100000,1000000,10000000]
    print(f"{'L':>12} | {'serial(s)':>12} | {'parallel (s)':>12}")
    for L in chunkSizes:
        tSerial, _ = granularityTest(numSamples,L,1)
        tParallel, pi = granularityTest(numSamples,L,nProcessors)
        print(f"{L:12d} | {tSerial:12.4f} | {tParallel:12.4f} | pi={pi:.4f}")