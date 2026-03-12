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
    samplesPerProcess = numSamples/processes
    jobs = [samplesPerProcess]*processes
    with Pool(processes=processes) as pool:
        results = pool.map(piSerial(numSamples), jobs)
    return 4*sum(results)/numSamples

if __name__ == '__main__':
    numSamples = 10000000
    for processors in range(1,os.cpu_count()+1):
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            #piEstimateSerial = piSerial(numSamples)
            piEstimateParallel = piParallel(numSamples)
            times.append(time.perf_counter() - t0)
        tSerial = statistics.median(times)
        print(f"{processors:2d} workers: {tSerial:.3f}s pi={piEstimateParallel:.6f}")
    print(f"pi estimate: {piEstimateSerial:.6f} (error: {abs(piEstimateS-math.pi):.6f})")
    print(f"Serial time: {tSerial:.3f}s")