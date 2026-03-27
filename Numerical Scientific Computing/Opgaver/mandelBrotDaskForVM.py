import numpy as np
import matplotlib.pyplot as plt
import time
import statistics
#from memory_profiler import memory_usage #wont work for some reason, and I dont wanna waste any more time on it
import numba
from numba import jit, njit, prange
import multiprocessing as mp
import cProfile, pstats
import os
import sys
import dask
from dask import delayed, compute
from dask.distributed import Client, LocalCluster
try:
    profile
except NameError:
    def profile(func):
        return func

testMemory = 0
gridSizes = [64,128,256,512,1024]#,2048,4096]
sizeComparison = 0
regions = {"Full": (-2, 1.5, -2, 2),"Seahorse Valley": (-0.8, -0.7, 0.05, 0.15),"Elephant Valley": (0.25, 0.35, -0.05, 0.05),"Deep Seahorse": (-0.7435, -0.7425, 0.1315, 0.1325)}
regionTest = 0
cProfiling = 0
makePlots = 0
gridPlot = 0
parallelScalingTest = 1

def main():
    print("Program begun")
    resultsForSummary = []

    if sizeComparison:
        for i in range(len(gridSizes)):
            gridSize = gridSizes[i]
            print(f"\n\n\nUsing gridsize {gridSize}")
            results = dataTypeComparison(gridSize)
            resultsForSummary.extend(results)
            if regionTest:
                regionComparison(gridSize)

            if parallelScalingTest:
                print("\nParallel scaling test")
                parallelResults = benchmarkParallel(gridSize, -2, 1.5, -2, 2, 50)
                ref = mandelbrotSerial(gridSize, -2, 1.5, -2, 2, 50)
                test = mandelbrotDask(gridSize, -2, 1.5, -2, 2, 50, nChunks=8)
                print("Match:", np.array_equal(ref, test))
                nChunksList = [1,2,4,8,16,32,64,125,256,512,1024]
                resultsDask, tSerial = benchmarkDask(gridSize, -2, 1.5, -2, 2, 50, nChunksList)
    else: 
        gridSize = 4096 # for speed in testing
        print(f"\n\n\nUsing gridsize {gridSize}")
        results = dataTypeComparison(gridSize)
        resultsForSummary.extend(results)
        if regionTest:
            regionComparison(gridSize)

        if parallelScalingTest:
            print("\nParallel scaling test")
            parallelResults = benchmarkParallel(gridSize, -2, 1.5, -2, 2, 50)
            ref = mandelbrotSerial(gridSize, -2, 1.5, -2, 2, 50)
            test = mandelbrotDask(gridSize, -2, 1.5, -2, 2, 50, nChunks=8)
            print("Match:", np.array_equal(ref, test))
            nChunksList = [1,2,4,8,16,32,64,125,256,512,1024]
            resultsDask, tSerial = benchmarkDask(gridSize, -2, 1.5, -2, 2, 50, nChunksList)
    

    if cProfiling: 
        print("cProfile testing initiated:")
        cProfile.run("mandelBrot(-2,1.5,-2,2,gridSize,np.float64)",'naiveProfile.prof')
        cProfile.run("mandelBrotFast(-2,1.5,-2,2,gridSize,np.float64)",'fastProfile.prof')
        for name in ('naiveProfile.prof', 'fastProfile.prof'):
            stats = pstats.Stats(name)
            stats.sort_stats('cumulative')
            stats.print_stats(10)
    else:
        print("cProfiling has not been done this run")

    if testMemory:
        memoryLayoutTest()
        print("\n")
        #memoryProfileComparison()
    else:
        print("Memory testing has not been done this run")

    if makePlots:
        if gridPlot:
            plt.tight_layout()
        plt.show()

    print("\n")
    summarize(resultsForSummary,parallelResults,resultsDask)
    printDaskTable(resultsDask, tSerial)
    #plotDask(resultsDask)

@profile
def mandelBrot(xMin, xMax, yMin, yMax, xyValues,dataType):
    xDomain, yDomain = np.linspace(xMin,xMax,xyValues).astype(dataType), np.linspace(yMin,yMax,xyValues).astype(dataType)
    bound = 2
    power = 2            # any positive floating point value (n)
    maxIterations = 50   # any positive integer value
    iterationArray = []

    for y in yDomain:
        row = []
        for x in xDomain:
            c = complex(x,y)
            z = 0
            for iterationNumber in range(maxIterations):
                if(abs(z) >= bound):
                    row.append(iterationNumber)
                    break
                else: z = z**power + c
            else:
                row.append(0)
        iterationArray.append(row)
    return np.array(iterationArray)

@profile
def mandelBrotFast(xMin, xMax, yMin, yMax, xyValues,dataType):
    xDomain, yDomain = np.linspace(xMin,xMax,xyValues).astype(dataType), np.linspace(yMin,yMax,xyValues).astype(dataType)
    bound = 2
    power = 2            # any positive floating point value (n)
    maxIterations = 50   # any positive integer value
    X, Y = np.meshgrid(xDomain,yDomain)
    C = X + 1j*Y
    #print(f"Shape: {C.shape}")
    #print(f"Type: {C.dtype}")
    # Z starts at 0 everywhere, M counts iterations
    Z = np.zeros_like(C, dtype=np.complex128)
    M = np.zeros(C.shape, dtype=np.int32)

    # Mandelbrot iterations (only loop we keep)
    for _ in range(maxIterations):
        mask = np.abs(Z) <= bound  # points not escaped yet
        # Update Z only where not escaped
        Z[mask] = Z[mask]**power + C[mask]
        # Count iterations only where not escaped
        M[mask] += 1
    M[M == maxIterations] = 0
    return M

def memoryLayoutTest(N=10000):
    print("Memory layout test")
    A = np.random.rand(N, N)

    def row_sum(A):
        total = 0.0
        for i in range(N):
            total += np.sum(A[i, :])
        return total

    def col_sum(A):
        total = 0.0
        for j in range(N):
            total += np.sum(A[:, j])
        return total

    # --- C order ---
    print("C-order array")
    print("A.flags:", A.flags['C_CONTIGUOUS'], A.flags['F_CONTIGUOUS'])
    t_row, _ = benchmark(row_sum, A)
    t_col, _ = benchmark(col_sum, A)
    print(f"Row loop time: {t_row:.4f}s")
    print(f"Col loop time: {t_col:.4f}s")

    # --- Fortran order ---
    A_f = np.asfortranarray(A)
    print("Fortran-order array")
    print("A_f.flags:", A_f.flags['C_CONTIGUOUS'], A_f.flags['F_CONTIGUOUS'])
    t_row_f, _ = benchmark(row_sum, A_f)
    t_col_f, _ = benchmark(col_sum, A_f)
    print(f"Row loop time (F-order): {t_row_f:.4f}s")
    print(f"Col loop time (F-order): {t_col_f:.4f}s")

@njit
def mandelBrotNumba(xMin, xMax, yMin, yMax, xyValues, dataType):
    bound = 2.0
    bound2 = bound * bound
    maxIterations = 50
    result = np.zeros((xyValues, xyValues), dtype=np.int32)

    for i in range(xyValues):
        y = yMin + (yMax - yMin) * i / (xyValues - 1)
        for j in range(xyValues):
            x = xMin + (xMax - xMin) * j / (xyValues - 1)
            zr = 0.0
            zi = 0.0
            for iterationNumber in range(maxIterations):
                if zr*zr + zi*zi >= bound2:
                    result[i, j] = iterationNumber
                    break
                # z = z^2 + c (manual complex arithmetic)
                temp = zr*zr - zi*zi + x
                zi = 2.0*zr*zi + y
                zr = temp
    return result

@njit(parallel=True)
def mandelBrotNumbaParallel(xMin, xMax, yMin, yMax, xyValues, dataType):
    bound = 2.0
    bound2 = bound * bound
    maxIterations = 50
    result = np.zeros((xyValues, xyValues), dtype=np.int32)

    for i in prange(xyValues):   # <- only change (parallel loop)
        y = yMin + (yMax - yMin) * i / (xyValues - 1)

        for j in range(xyValues):
            x = xMin + (xMax - xMin) * j / (xyValues - 1)
            zr = 0.0
            zi = 0.0
            for iterationNumber in range(maxIterations):
                if zr*zr + zi*zi >= bound2:
                    result[i, j] = iterationNumber
                    break
                # z = z^2 + c (manual complex arithmetic)
                temp = zr*zr - zi*zi + x
                zi = 2.0 * zr * zi + y
                zr = temp

    return result

@njit
def mandelbrotPixel(cReal, cImag, maxIter):
    zr =0.0
    zi = 0.0
    bound2 =4.0
    for i in range(maxIter):
        if zr*zr+zi*zi>=bound2:
            return i
        temp = zr*zr-zi*zi+cReal
        zi = 2.0*zr*zi+cImag
        zr=temp
    return 0

@njit#(cache=True)
def mandelbrotChunk(rowStart, rowEnd, N, xMin, xMax, yMin, yMax, maxIter):
    result = np.zeros((rowEnd - rowStart, N), dtype=np.int32)
    for i in range(rowStart, rowEnd):
        dy = (yMax - yMin) / (N - 1)
        y = yMin + i * dy
        #y = yMin + (yMax - yMin) * i / (N - 1)
        for j in range(N):
            dx = (xMax - xMin) / (N - 1)
            x = xMin + j * dx
#            x = xMin + (xMax - xMin) * j / (N - 1)
            result[i - rowStart, j] = mandelbrotPixel(x, y, maxIter)
    return result

def mandelbrotSerial(N, xMin, xMax, yMin, yMax, maxIter):
    return mandelbrotChunk(0,N,N,xMin,xMax,yMin,yMax,maxIter)

def worker(args):
    return mandelbrotChunk(*args)

def mandelbrotParallel(N, xMin, xMax, yMin, yMax, maxIter, nWorkers):
    rowsPerChunk = N // nWorkers
    chunks = []
    for w in range(nWorkers):
        rowStart = w * rowsPerChunk
        if w == nWorkers - 1:
            rowEnd = N
        else:
            rowEnd = (w + 1) * rowsPerChunk
        chunks.append(
            (rowStart, rowEnd, N, xMin, xMax, yMin, yMax, maxIter)
        )
    with mp.Pool(processes=nWorkers) as pool:
        parts = pool.map(worker, chunks)
    result = np.vstack(parts)
    return result

def benchmarkParallel(N, xMin, xMax, yMin, yMax, maxIter):
    print("\nRunning parallel sweep")
    results = []
    mandelbrotSerial(N, xMin, xMax, yMin, yMax, maxIter)  # warmup
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrotSerial(N, xMin, xMax, yMin, yMax, maxIter)
        times.append(time.perf_counter() - t0)
    tSerial = statistics.median(times)
    print(f"Serial median: {tSerial:.4f}s")
    cpuCount = os.cpu_count()
    for p in range(1, cpuCount + 1):
        print(f"\nWorkers: {p}")
        rowsPerChunk = N // p
        chunks = []
        for w in range(p):
            rowStart = w * rowsPerChunk
            if w == p - 1:
                rowEnd = N
            else:
                rowEnd = (w + 1) * rowsPerChunk
            chunks.append((rowStart, rowEnd, N, xMin, xMax, yMin, yMax, maxIter))
        with mp.Pool(processes=p) as pool:
            # Warmup run (forces Numba compilation inside workers)
            pool.map(worker, chunks)
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                parts = pool.map(worker, chunks)
                result = np.vstack(parts)
                times.append(time.perf_counter() - t0)
        tp = statistics.median(times)
        speedup = tSerial / tp
        efficiency = speedup / p
        print(f"time: {tp:.4f}s")
        print(f"speedup: {speedup:.2f}")
        print(f"efficiency: {efficiency:.2f}")
        results.append({
            "gridSize": N,
            "workers": p,
            "tSerial": tSerial,
            "tParallel": tp,
            "speedup": speedup,
            "efficiency": efficiency
        })
    return results

def warmupWorkers(client):
    def warmup():
        mandelbrotChunk(0, 4, 8, -2, 1.5, -2, 2, 10)
    client.run(warmup)

def mandelbrotDask(N, xMin, xMax, yMin, yMax, maxIter, nChunks):
    rowsPerChunk = N // nChunks
    tasks = []

    for w in range(nChunks):
        rowStart = w * rowsPerChunk
        if w == nChunks - 1:
            rowEnd = N
        else:
            rowEnd = (w + 1) * rowsPerChunk

        task = delayed(mandelbrotChunk)(
            rowStart, rowEnd, N, xMin, xMax, yMin, yMax, maxIter
        )
        tasks.append(task)

    results = compute(*tasks)
    return np.vstack(results)

def benchmarkDask(N, xMin, xMax, yMin, yMax, maxIter, nChunksList):
    print("\nDask chunk sweep")

    results = []

    #cluster = LocalCluster(processes=True, threads_per_worker=1)
    #client = Client(cluster)
    client = Client("tcp://10.92.1.120:8786")
    #print(f"Dashboard: {client.dashboard_link}")

    warmupWorkers(client)

    # Serial baseline
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrotSerial(N, xMin, xMax, yMin, yMax, maxIter)
        times.append(time.perf_counter() - t0)
    tSerial = statistics.median(times)

    print(f"Serial time: {tSerial:.4f}s")

    for nChunks in nChunksList:
        print(f"\nChunks: {nChunks}")

        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            mandelbrotDask(N, xMin, xMax, yMin, yMax, maxIter, nChunks)
            times.append(time.perf_counter() - t0)

        tParallel = statistics.median(times)

        p = os.cpu_count()
        speedup = tSerial / tParallel
        lif = p * (tParallel / tSerial) - 1
        vs1x = tParallel / tSerial

        print(f"time: {tParallel:.4f}s")
        print(f"speedup: {speedup:.2f}")
        print(f"LIF: {lif:.2f}")

        results.append({"nChunks": nChunks,"time": tParallel,"speedup": speedup,"lif": lif,"vs1x": vs1x})

    client.close()
    #cluster.close()

    return results, tSerial

def plotDask(results):
    chunks = [r["nChunks"] for r in results]
    times = [r["time"] for r in results]
    plt.figure()
    plt.plot(chunks, times, marker='o')
    plt.xscale('log')
    plt.xlabel("nChunks (log scale)")
    plt.ylabel("Time (s)")
    plt.title("Dask chunk sweep")
    plt.grid(True)
    plt.show()
    #plt.savefig("dask_chunk_sweep.png")

def printDaskTable(results, tSerial):
    print("\nn chunks | time (s) | vs 1x | speedup | Load Imbalance Factor")
    print("-" * 50)

    for r in results:
        vs1x = r["time"] / tSerial
        print(f"{r['nChunks']:8} | {r['time']:8.4f} | {vs1x:6.2f} | {r['speedup']:7.2f} | {r['lif']:6.2f}")

def getBestMP(parallelResults):
    best = min(parallelResults, key=lambda x: x["tParallel"])
    return best


def getBestDask(resultsDask):
    best = min(resultsDask, key=lambda x: x["time"])
    return best


def benchmark(func, *args, nRuns=10):
    times = []

    for _ in range (nRuns):
        startTime = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter()-startTime)
    
    medianTime = statistics.median(times)
    print(f"Median: {medianTime:.4f}s, Mean: {statistics.mean(times):.4f}, Min: {min(times):.4f}, Max: {max(times):.4f}")
    return medianTime, result

def plotMandelbrot(result, xMin, xMax, yMin, yMax,title="Mandelbrot", ax=None):
    xDomain = np.linspace(xMin, xMax, result.shape[1])
    yDomain = np.linspace(yMin, yMax, result.shape[0])

    if ax is None:
        fig, ax = plt.subplots()
    
    mesh = ax.pcolormesh(xDomain, yDomain, result,cmap="magma", shading="auto")
    ax.set_xlabel("Real-Axis")
    ax.set_ylabel("Imaginary-Axis")
    ax.set_title(title)
    ax.set_aspect("equal")
    return mesh

def dataTypeComparison(gridSize):
    results = []
    if makePlots and gridPlot:
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for row, dType in enumerate([np.float32]):#[np.float16, np.float32, np.float64]): # For fun add: , np.int8, np.int16, np.int32]:
        mandelBrotNumba(-2,1.5,-2,2,64,dType) # Warmup
        print(f"Numba warmed up using {dType.__name__}.")
        print(f"Benchmarking begins for {dType.__name__}:")
        medianSlow, resultSlow = benchmark(mandelBrot,-2,1.5,-2,2,gridSize,dType)
        medianFast, resultFast = benchmark(mandelBrotFast,-2,1.5,-2,2,gridSize,dType)
        medianNumba, resultNumba = benchmark(mandelBrotNumba,-2,1.5,-2,2,gridSize,dType)
        medianNumbaParallel, resultNumbaParallel = benchmark(mandelBrotNumbaParallel,-2,1.5,-2,2,gridSize,dType)

        if np.allclose(resultSlow,resultFast) and np.allclose(resultSlow,resultNumba):
            print("Results match!")
        else:
            print("Results differ!")

        # Check where they differ :
        diffSF = np.abs(resultSlow - resultFast)
        print(f"Max difference between slow and fast: {diffSF.max ()}")
        print(f"Different pixels between slow and fast: {(diffSF>0).sum()}")
        diffSN = np.abs(resultSlow - resultNumba)
        print(f"Max difference between slow and numba: {diffSN.max ()}")
        print(f"Different pixels between slow and numba: {(diffSN>0).sum()}")
        diffFN = np.abs(resultFast - resultNumba)
        print(f"Max difference between fast and numba: {diffFN.max ()}")
        print(f"Different pixels between fast and numba: {(diffFN>0).sum()}")
        
        # Speedup
        speedup = medianSlow / medianFast
        print(f"Speedup (medianSlow / medianFast): {speedup:.2f}x")
        speedup = medianSlow / medianNumba
        print(f"Speedup (medianSlow / medianNumba): {speedup:.2f}x")
        speedup = medianFast / medianNumba
        print(f"Speedup (medianFast / medianNumba): {speedup:.2f}x")
        speedup = medianSlow / medianNumbaParallel
        print(f"Speedup (medianSlow / medianNumbaParallel): {speedup:.2f}x")
        speedup = medianFast / medianNumbaParallel
        print(f"Speedup (medianFast / medianNumbaParallel): {speedup:.2f}x")
        speedup = medianNumba / medianNumbaParallel
        print(f"Speedup (medianNumba / medianNumbaParallel): {speedup:.2f}x")
        
        if makePlots and gridPlot:
            print("Preparing plots")
            plotMandelbrot(resultSlow, -2, 1.5, -2, 2,f"Slow ({dType.__name__})",ax=axes[row, 0])
            plotMandelbrot(resultFast, -2, 1.5, -2, 2,f"Fast ({dType.__name__})",ax=axes[row, 1])
            plotMandelbrot(resultNumba, -2, 1.5, -2, 2,f"Numba ({dType.__name__})",ax=axes[row, 2])
        elif makePlots:
            print("Preparing plots")
            plotMandelbrot(resultSlow, -2, 1.5, -2, 2, f"Slow Version using {dType.__name__}")
            plotMandelbrot(resultFast, -2, 1.5, -2, 2, f"Fast Version using {dType.__name__}")
            plotMandelbrot(resultNumba, -2, 1.5, -2, 2, f"Numba Version using {dType.__name__}")
            plotMandelbrot(resultNumbaParallel, -2, 1.5, -2, 2, f"Numba Version using {dType.__name__}")
        else:
            print("No plots made")
        
        entry = {
        "gridSize": gridSize,
        "dtype": dType.__name__,

        "medianSlow": medianSlow,
        "medianFast": medianFast,
        "medianNumba": medianNumba,
        "medianNumbaParallel": medianNumbaParallel,

        "speedupSlowFast": medianSlow / medianFast,
        "speedupSlowNumba": medianSlow / medianNumba,
        "speedupFastNumba": medianFast / medianNumba,
        "speedupSlowNumbaParallel": medianSlow / medianNumbaParallel,
        "speedupFastNumbaParallel": medianFast / medianNumbaParallel,
        "speedupNumbaNumbaParallel": medianNumba / medianNumbaParallel,

        "maxDiffSlowFast": diffSF.max(),
        "diffPixelsSlowFast": (diffSF > 0).sum(),

        "maxDiffSlowNumba": diffSN.max(),
        "diffPixelsSlowNumba": (diffSN > 0).sum(),

        "maxDiffFastNumba": diffFN.max(),
        "diffPixelsFastNumba": (diffFN > 0).sum(),
        }
        results.append(entry)
    return results

def summarize(results, resultsParallel,resultsDask):
    print("In summary:")
    bestMP = getBestMP(resultsParallel) if resultsParallel else None
    bestDask = getBestDask(resultsDask) if resultsDask else None
    tMP = bestMP["tParallel"] if bestMP else None
    tDask = bestDask["time"] if bestDask else None


    header = (f"{'Grid':>6} | {'DType':>8} | "f"{'Slow':>8} | {'Fast':>8} | {'Numba':>8} | {'NumbaParallel':>14} | "f"{'S/F':>6} | {'S/N':>6} | {'F/N':>6} | {'S/NP':>6} | {'F/NP':>6} | {'N/NP':>6} | {'MP/S':>6} | {'MP/F':>6} | {'MP/N':>6} | {'MP/NP':>6} | {'D/S':>6} | {'D/F':>6} | {'D/N':>6} | {'D/NP':>6}")
    print(header)
    print("-" * len(header))
    for r in results:
        tSlow = r["medianSlow"]
        tFast = r["medianFast"]
        tNumba = r["medianNumba"]
        tNumbaParallel = r["medianNumbaParallel"]

        # MP speedups
        mp_s = tSlow / tMP if tMP else 0
        mp_f = tFast / tMP if tMP else 0
        mp_n = tNumba / tMP if tMP else 0
        mp_np = tNumbaParallel / tMP if tMP else 0

        # Dask speedups
        d_s = tSlow / tDask if tDask else 0
        d_f = tFast / tDask if tDask else 0
        d_n = tNumba / tDask if tDask else 0
        d_np = tNumbaParallel / tDask if tDask else 0
        print(f"{r['gridSize']:6} | "f"{r['dtype']:8} | "f"{r['medianSlow']:8.4f} | "f"{r['medianFast']:8.4f} | "f"{r['medianNumba']:8.4f} | "f"{r['medianNumbaParallel']:14.4f} | "f"{r['speedupSlowFast']:6.2f} | "f"{r['speedupSlowNumba']:6.2f} | "f"{r['speedupFastNumba']:6.2f} | "f"{r['speedupSlowNumbaParallel']:6.2f} | "f"{r['speedupFastNumbaParallel']:6.2f} | "f"{r['speedupNumbaNumbaParallel']:6.2f} | {mp_s:6.2f} | {mp_f:6.2f} | {mp_n:6.2f} | {mp_np:6.2f} | {d_s:6.2f} | {d_f:6.2f} | {d_n:6.2f} | {d_np:6.2f}")
    
    parallelData = resultsParallel
    if parallelData:
        print("\nParallel scaling summary:")
        header = (f"{'Grid':>6} | {'Workers':>8} | "f"{'tSerial':>10} | {'tParallel':>10} | "f"{'Speedup':>8} | {'Efficiency':>10}")
        print(header)
        print("-" * len(header))
        for r in parallelData:
            print(f"{r['gridSize']:6} | "f"{r['workers']:8} | "f"{r['tSerial']:10.4f} | "f"{r['tParallel']:10.4f} | "f"{r['speedup']:8.2f} | "f"{r['efficiency']:10.2f}")

def memoryProfileComparison(gridSize=1024):
    print("Making memory profiling")

    def run_naive():
        mandelBrot(-2, 1.5, -2, 2, gridSize, np.float64)

    def run_fast():
        mandelBrotFast(-2, 1.5, -2, 2, gridSize, np.float64)

    # Measure peak memory usage
    mem_naive = memory_usage(run_naive, max_usage=True)
    mem_fast = memory_usage(run_fast, max_usage=True)

    print(f"Naive peak memory: {mem_naive:.2f} MiB")
    print(f"Vectorized peak memory: {mem_fast:.2f} MiB")

def regionComparison(gridSize=1024):
    print("Testing different regions")

    for name, (xMin, xMax, yMin, yMax) in regions.items():
        print(f"Region: {name}")
        tSlow, _ = benchmark(mandelBrot,xMin, xMax, yMin, yMax,gridSize, np.float64)
        tFast, _ = benchmark(mandelBrotFast,xMin, xMax, yMin, yMax,gridSize, np.float64)
        tNumba, _ = benchmark(mandelBrotNumba,xMin, xMax, yMin, yMax,gridSize, np.float64)
        print(f"Slow:  {tSlow:.4f}s")
        print(f"Fast:  {tFast:.4f}s")
        print(f"Numba: {tNumba:.4f}s")

if __name__ == '__main__':
    main()