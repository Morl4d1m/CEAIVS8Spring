import numpy as np
import matplotlib.pyplot as plt
import time
import statistics
from memory_profiler import memory_usage #wont work for some reason, and I dont wanna waste any more time on it
#import numba
from numba import njit, prange#, jit
import multiprocessing as mp
import cProfile
import pstats
import os
#import sys
#import dask
from dask import delayed, compute
from dask.distributed import Client, LocalCluster
try:
    profile
except NameError:
    def profile(func):
        return func
from matplotlib.colors import LogNorm
import pyopencl as cl

testMemory = 1
gridSizes = [64,128,256,512,1024,2048,4096]
sizeComparison = 1
regions = {"Full": (-2, 1.5, -2, 2),"Seahorse Valley": (-0.8, -0.7, 0.05, 0.15),"Elephant Valley": (0.25, 0.35, -0.05, 0.05),"Deep Seahorse": (-0.7435, -0.7425, 0.1315, 0.1325)}
regionTest = 0
cProfiling = 0
makePlots = 0
gridPlot = 0
parallelScalingTest = 1
checkEps = 0
checkPrecision = 0
runTests = 0
useOpenCL = 1

kernelF32 = """
__kernel void mandelbrot_f32(
    __global int *result,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= N || row >= N) return;

    float c_real = x_min + col * (x_max - x_min) / (float)N;
    float c_imag = y_min + row * (y_max - y_min) / (float)N;

    float z_real = 0.0f, z_imag = 0.0f;
    int count = 0;
    while (count < max_iter && z_real*z_real + z_imag*z_imag <= 4.0f) {
        float tmp = z_real*z_real - z_imag*z_imag + c_real;
        z_imag = 2.0f * z_real * z_imag + c_imag;
        z_real = tmp;
        count++;
    }
    result[row * N + col] = count;
}
"""

kernelF64 = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void mandelbrot_f64(
    __global int *result,
    const double x_min, const double x_max,
    const double y_min, const double y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= N || row >= N) return;

    double c_real = x_min + col * (x_max - x_min) / (double)N;
    double c_imag = y_min + row * (y_max - y_min) / (double)N;

    double z_real = 0.0, z_imag = 0.0;
    int count = 0;
    while (count < max_iter && z_real*z_real + z_imag*z_imag <= 4.0) {
        double tmp = z_real*z_real - z_imag*z_imag + c_real;
        z_imag = 2.0 * z_real * z_imag + c_imag;
        z_real = tmp;
        count++;
    }
    result[row * N + col] = count;
}
"""

def main():
    print("Program begun")
    resultsForSummary = []
    if runTests:
        runUnitTests()
        return

    if useOpenCL:
        openclResults = runOpenCLBenchmark()
        #return # for debugging opencl
    else:
        openclResults = None

    if checkEps:
        for dtype in [np.float16,np.float32,np.float64]:
            computed = find_machine_epsilon(dtype)
            reference = np.finfo(dtype).eps
            print(f"{dtype.__name__}:")
            print(f"Computed: {float(computed):.4e}")
            print(f"finfo: {float(reference):.4e}")
        
        for dtype in [np.float32, np.float64]:
            a, b, c = dtype(1.0), dtype(-10000.0001), dtype(1.0)
            x1, x2 = quadratic_naive(a, b, c)
            print(f"{dtype.__name__}: x1 = {float(x1):.4f}, x2 = {float(x2):.10f}")
        
        true_small = 1.0 / 10000.0001 # ~ 1e-4
        for dtype in [np.float32, np.float64]:
            a, b, c = dtype(1.0), dtype(-10000.0001), dtype(1.0)
            _, x2_naive = quadratic_naive(a, b, c)
            _, x2_stable = quadratic_stable(a, b, c)
            err_naive = abs(float(x2_naive) - true_small) / true_small
            err_stable = abs(float(x2_stable) - true_small) / true_small
            print(f"{dtype.__name__}: naive={err_naive:.2e} stable={err_stable:.2e}")

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
        gridSize = 1024 # for speed in testing
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
    summarize(resultsForSummary,parallelResults,resultsDask,openclResults)
    printDaskTable(resultsDask, tSerial)
    #plotDask(resultsDask)
    if checkPrecision:
        mandelbrotDivergenceMap() #divergence 
        mandelbrotConditionMap() #kappa 

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
                else: 
                    z = z**power + c
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
def mandelbrotPixel(cReal, cImag, maxIter) -> int:
    """
    Compute Mandelbrot escape iteration for a single complex point.

    Parameters
    ----------
    cReal : float
        Real part of the complex parameter c.
    cImag : float
        Imaginary part of the complex parameter c.
    maxIter : int
        Maximum number of iterations before declaring the point bounded.

    Returns
    -------
    int
        Iteration index at which |z| >= 2. Returns 0 if the point does not
        escape within maxIter iterations.
    """
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
def mandelbrotChunk(rowStart, rowEnd, N, xMin, xMax, yMin, yMax, maxIter) -> np.ndarray:
    """
    Compute a row chunk of the Mandelbrot grid.

    This function evaluates the Mandelbrot escape iteration for rows
    [rowStart, rowEnd) of an N x N grid. It is designed for use in
    multiprocessing and distributed execution.

    Parameters
    ----------
    rowStart : int
        First row index (inclusive).
    rowEnd : int
        Last row index (exclusive).
    N : int
        Grid resolution (NxN).
    xMin : float
        Minimum real-axis value.
    xMax : float
        Maximum real-axis value.
    yMin : float
        Minimum imaginary-axis value.
    yMax : float
        Maximum imaginary-axis value.
    maxIter : int
        Maximum number of iterations.

    Returns
    -------
    np.ndarray
        2D array of shape (rowEnd-rowStart, N) containing iteration counts.
    """
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

def mandelbrotSerial(N, xMin, xMax, yMin, yMax, maxIter) -> np.ndarray:
    """
    Compute Mandelbrot set using a serial implementation.

    Parameters
    ----------
    N : int
        Grid resolution (NxN).
    xMin : float
        Minimum real-axis value.
    xMax : float
        Maximum real-axis value.
    yMin : float
        Minimum imaginary-axis value.
    yMax : float
        Maximum imaginary-axis value.
    maxIter : int
        Maximum number of iterations.

    Returns
    -------
    np.ndarray
        NxN array of Mandelbrot escape iterations.
    """
    return mandelbrotChunk(0,N,N,xMin,xMax,yMin,yMax,maxIter)

def worker(args):
    return mandelbrotChunk(*args)

def mandelbrotParallel(N, xMin, xMax, yMin, yMax, maxIter, nWorkers) -> np.ndarray:
    """
    Compute Mandelbrot set using multiprocessing.

    The grid is divided into row chunks distributed across worker
    processes. Each worker computes its chunk independently.

    Parameters
    ----------
    N : int
        Grid resolution (NxN).
    xMin : float
        Minimum real-axis value.
    xMax : float
        Maximum real-axis value.
    yMin : float
        Minimum imaginary-axis value.
    yMax : float
        Maximum imaginary-axis value.
    maxIter : int
        Maximum number of iterations.
    nWorkers : int
        Number of multiprocessing workers.

    Returns
    -------
    np.ndarray
        NxN Mandelbrot escape iteration grid.
    """
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
                np.vstack(parts)
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

def mandelbrotDask(N, xMin, xMax, yMin, yMax, maxIter, nChunks) -> np.ndarray:
    """
    Compute Mandelbrot set using Dask delayed execution.

    The grid is divided into chunks which are scheduled as
    independent Dask tasks.

    Parameters
    ----------
    N : int
        Grid resolution (NxN).
    xMin : float
        Minimum real-axis value.
    xMax : float
        Maximum real-axis value.
    yMin : float
        Minimum imaginary-axis value.
    yMax : float
        Maximum imaginary-axis value.
    maxIter : int
        Maximum number of iterations.
    nChunks : int
        Number of Dask chunks.

    Returns
    -------
    np.ndarray
        NxN Mandelbrot escape iteration grid.
    """
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

    cluster = LocalCluster(processes=True, threads_per_worker=1)
    client = Client(cluster)
    print(f"Dashboard: {client.dashboard_link}")

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
    cluster.close()

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

def summarize(results, resultsParallel,resultsDask, openclResults=None):
    print("In summary:")
    bestMP = getBestMP(resultsParallel) if resultsParallel else None
    bestDask = getBestDask(resultsDask) if resultsDask else None
    tMP = bestMP["tParallel"] if bestMP else None
    tDask = bestDask["time"] if bestDask else None


    header = (f"{'Grid':>6} | {'DType':>8} | "f"{'Slow':>8} | {'Fast':>8} | {'Numba':>8} | {'NumbaParallel':>14} | "f"{'S/F':>6} | {'S/N':>6} | {'F/N':>6} | {'S/NP':>6} | {'F/NP':>6} | {'N/NP':>6} | {'MP/S':>6} | {'MP/F':>6} | {'MP/N':>6} | {'MP/NP':>6} | {'D/S':>6} | {'D/F':>6} | {'D/N':>6} | {'D/NP':>6} | {'CL32':>8} | {'CL64':>8} | {'CL64/32':>8} | {'CL32/S':>8} | {'CL32/N':>8} | {'CL32/NP':>8} | {'CL64/S':>8} | {'CL64/N':>8} | {'CL64/NP':>8}")
    print(header)
    print("-" * len(header))
    openclMap = {}
    if openclResults:
        for r in openclResults:
            openclMap[r["gridSize"]] = r
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

        #OpenCL
        cl32 = openclMap.get(r["gridSize"], {}).get("opencl32", 0)
        cl64 = openclMap.get(r["gridSize"], {}).get("opencl64", 0)
        clratio = openclMap.get(r["gridSize"], {}).get("ratio", 0)

        # OpenCL speedups
        cl32_s = tSlow / cl32 if cl32 else 0
        cl32_n = tNumba / cl32 if cl32 else 0
        cl32_np = tNumbaParallel / cl32 if cl32 else 0
        cl64_s = tSlow / cl64 if cl64 else 0
        cl64_n = tNumba / cl64 if cl64 else 0
        cl64_np = tNumbaParallel / cl64 if cl64 else 0


        print(f"{r['gridSize']:6} | "f"{r['dtype']:8} | "f"{r['medianSlow']:8.4f} | "f"{r['medianFast']:8.4f} | "f"{r['medianNumba']:8.4f} | "f"{r['medianNumbaParallel']:14.4f} | "f"{r['speedupSlowFast']:6.2f} | "f"{r['speedupSlowNumba']:6.2f} | "f"{r['speedupFastNumba']:6.2f} | "f"{r['speedupSlowNumbaParallel']:6.2f} | "f"{r['speedupFastNumbaParallel']:6.2f} | "f"{r['speedupNumbaNumbaParallel']:6.2f} | {mp_s:6.2f} | {mp_f:6.2f} | {mp_n:6.2f} | {mp_np:6.2f} | {d_s:6.2f} | {d_f:6.2f} | {d_n:6.2f} | {d_np:6.2f} | {cl32:8.4f} | {cl64:8.4f} | {clratio:8.2f} | {cl32_s:8.2f} | {cl32_n:8.2f} | {cl32_np:8.2f} | {cl64_s:8.2f} | {cl64_n:8.2f} | {cl64_np:8.2f}")
    
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

def find_machine_epsilon(dtype=np.float64):
    eps = dtype(1.0)
    while dtype(1.0) + eps / dtype(2.0) != dtype(1.0):
        eps = eps / dtype(2.0)
    return eps

def quadratic_naive(a, b, c):
    t = type(a) # np.float32 or np.float64
    disc = t(np.sqrt(b*b - t(4)*a*c)) # b*b not b**2; t() casts literals and sqrt
    x1 = (-b + disc) / (t(2)*a)
    x2 = (-b - disc) / (t(2)*a)
    return x1, x2

def quadratic_stable(a, b, c):
    t = type(a)
    disc = t(np.sqrt(b*b - t(4)*a*c))
    if b > 0:
        x1 = (-b - disc) / (t(2)*a) # pick sign that avoids cancellation
    else:
        x1 = (-b + disc) / (t(2)*a)
    x2 = c / (a * x1) # Vieta’s formula: x1 * x2 = c/a
    return x1, x2

def mandelbrotDivergenceMap(
    N=512,
    maxIter=1000,
    tau=0.01,
    xMin=-0.7530, xMax=-0.7490,
    yMin=0.0990, yMax=0.1030
):
    print("\nRunning M1: Float32 vs Float64 divergence map")

    x = np.linspace(xMin, xMax, N)
    y = np.linspace(yMin, yMax, N)

    C64 = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)
    C32 = C64.astype(np.complex64)

    z32 = np.zeros_like(C32)
    z64 = np.zeros_like(C64)

    diverge = np.full((N, N), maxIter, dtype=np.int32)
    active = np.ones((N, N), dtype=bool)

    for k in range(maxIter):
        if not active.any():
            break

        z32[active] = z32[active]**2 + C32[active]
        z64[active] = z64[active]**2 + C64[active]

        diff = (
            np.abs(z32.real.astype(np.float64) - z64.real)
            + np.abs(z32.imag.astype(np.float64) - z64.imag)
        )

        newly = active & (diff > tau)
        diverge[newly] = k
        active[newly] = False

    # fraction diverging before maxIter
    frac = np.mean(diverge < maxIter)
    print(f"Fraction diverging before maxIter: {frac:.3f}")

    plt.figure()
    plt.imshow(
        diverge,
        cmap="plasma",
        origin="lower",
        extent=[xMin, xMax, yMin, yMax]
    )
    plt.colorbar(label="First divergence iteration")
    plt.title(f"Float32 vs Float64 divergence (tau={tau})")
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")
    plt.show()

    return diverge

def mandelbrotConditionMap(
    N=512,
    maxIter=1000,
    xMin=-0.7530, xMax=-0.7490,
    yMin=0.0990, yMax=0.1030
):
    print("\nRunning M2: Mandelbrot sensitivity map")

    x = np.linspace(xMin, xMax, N)
    y = np.linspace(yMin, yMax, N)

    C = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)

    eps32 = float(np.finfo(np.float32).eps)
    delta = np.maximum(eps32 * np.abs(C), 1e-10)

    def escapeCount(C, maxIter):
        z = np.zeros_like(C)
        cnt = np.full(C.shape, maxIter, dtype=np.int32)
        escaped = np.zeros(C.shape, dtype=bool)

        for k in range(maxIter):
            z[~escaped] = z[~escaped]**2 + C[~escaped]

            newly = ~escaped & (np.abs(z) > 2.0)
            cnt[newly] = k
            escaped[newly] = True

        return cnt

    print("Computing base escape counts...")
    nBase = escapeCount(C, maxIter).astype(float)

    print("Computing perturbed escape counts...")
    nPerturb = escapeCount(C + delta, maxIter).astype(float)

    dn = np.abs(nBase - nPerturb)

    kappa = np.where(
        nBase > 0,
        dn / (eps32 * nBase),
        np.nan
    )

    cmap = plt.cm.hot.copy()
    cmap.set_bad("0.25")

    vmax = np.nanpercentile(kappa, 99)

    plt.figure()
    plt.imshow(
        kappa,
        cmap=cmap,
        origin="lower",
        extent=[xMin, xMax, yMin, yMax],
        norm=LogNorm(vmin=1, vmax=vmax)
    )

    plt.colorbar(label=r'$\kappa(c)$ (log scale)')
    plt.title("Mandelbrot condition number approximation")
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")
    plt.show()

    return kappa

def runUnitTests():
    print("\nRunning Mandelbrot tests\n")

    testPixelKnownValues()
    testChunkMatchesSerial()
    testParallelMatchesSerial()
    testDaskMatchesSerial()
    testImplementationsAgree()

    print("\nAll tests passed\n")

def testPixelKnownValues():
    print("Test: pixel known values")

    assert mandelbrotPixel(0.0, 0.0, 50) == 0
    assert mandelbrotPixel(2.0, 0.0, 50) == 1
    assert mandelbrotPixel(-2.0, 0.0, 50) == 1

    # known diverging point
    val = mandelbrotPixel(0.5, 0.5, 50)
    assert val > 0

    print("  passed")

def testChunkMatchesSerial():
    print("Test: chunk vs serial")

    N = 32
    args = (-2, 1.5, -2, 2, 50)

    serial = mandelbrotSerial(N, *args)
    chunk = mandelbrotChunk(0, N, N, *args)

    assert np.array_equal(serial, chunk)

    print("  passed")

def testParallelMatchesSerial():
    print("Test: multiprocessing vs serial")

    N = 32
    args = (-2, 1.5, -2, 2, 50)

    serial = mandelbrotSerial(N, *args)
    parallel = mandelbrotParallel(N, *args, nWorkers=2)

    assert np.array_equal(serial, parallel)

    print("  passed")

def testDaskMatchesSerial():
    print("Test: dask vs serial")

    N = 32
    args = (-2, 1.5, -2, 2, 50)

    serial = mandelbrotSerial(N, *args)
    dask2 = mandelbrotDask(N, *args, nChunks=4)

    assert np.array_equal(serial, dask2)

    print("  passed")

def testImplementationsAgree():
    print("Test: implementations agree")

    N = 32

    slow = mandelBrot(-2, 1.5, -2, 2, N, np.float64)
    fast = mandelBrotFast(-2, 1.5, -2, 2, N, np.float64)
    numba2 = mandelBrotNumba(-2, 1.5, -2, 2, N, np.float64)

    assert np.array_equal(slow, fast)
    assert np.array_equal(slow, numba2)

    print("  passed")

def initOpenCL():
    """
    Initialize OpenCL context and queue.

    Returns
    -------
    tuple[cl.Context, cl.CommandQueue]
        OpenCL context and command queue.
    """
    import pyopencl as cl

    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    device = ctx.devices[0]
    print(f"OpenCL device: {device.name}")

    return ctx, queue


def mandelbrotOpenCL32(
    ctx,
    queue,
    N: int,
    xMin: float,
    xMax: float,
    yMin: float,
    yMax: float,
    maxIter: int,
) -> np.ndarray:
    """
    Compute Mandelbrot set using OpenCL float32 kernel.

    Parameters
    ----------
    ctx : cl.Context
        OpenCL context.
    queue : cl.CommandQueue
        OpenCL command queue.
    N : int
        Grid resolution.
    xMin, xMax, yMin, yMax : float
        Domain limits.
    maxIter : int
        Maximum iterations.

    Returns
    -------
    np.ndarray
        Mandelbrot iteration grid.
    """

    program = cl.Program(ctx, kernelF32).build()

    resultHost = np.zeros((N, N), dtype=np.int32)
    resultDev = cl.Buffer(
        ctx,
        cl.mem_flags.WRITE_ONLY,
        resultHost.nbytes
    )

    program.mandelbrot_f32(
        queue,
        (N, N),
        None,
        resultDev,
        np.float32(xMin),
        np.float32(xMax),
        np.float32(yMin),
        np.float32(yMax),
        np.int32(N),
        np.int32(maxIter),
    )

    cl.enqueue_copy(queue, resultHost, resultDev)
    queue.finish()

    return resultHost

def mandelbrotOpenCL64(
    ctx,
    queue,
    N: int,
    xMin: float,
    xMax: float,
    yMin: float,
    yMax: float,
    maxIter: int,
) -> np.ndarray | None:
    """
    Compute Mandelbrot set using OpenCL float64 kernel.

    Returns None if device does not support fp64.
    """

    device = ctx.devices[0]

    if "cl_khr_fp64" not in device.extensions:
        print("Float64 not supported on device")
        return None

    program = cl.Program(ctx, kernelF64).build()

    resultHost = np.zeros((N, N), dtype=np.int32)
    resultDev = cl.Buffer(
        ctx,
        cl.mem_flags.WRITE_ONLY,
        resultHost.nbytes
    )

    program.mandelbrot_f64(
        queue,
        (N, N),
        None,
        resultDev,
        np.float64(xMin),
        np.float64(xMax),
        np.float64(yMin),
        np.float64(yMax),
        np.int32(N),
        np.int32(maxIter),
    )

    cl.enqueue_copy(queue, resultHost, resultDev)
    queue.finish()

    return resultHost

def runOpenCLBenchmark():
    print("\nRunning OpenCL benchmark")

    ctx, queue = initOpenCL()

    maxIter = 100
    Ns = [1024, 2048, 4096]

    results = []

    # warmup compile
    mandelbrotOpenCL32(ctx, queue, 64, -2, 1.5, -2, 2, maxIter)

    for N in Ns:
        print(f"\nN = {N}")

        t32, img32 = benchmark(
            mandelbrotOpenCL32,
            ctx, queue,
            N, -2, 1.5, -2, 2,
            maxIter
        )

        print(f"OpenCL float32: {t32:.4f}s")

        img64 = mandelbrotOpenCL64(
            ctx, queue,
            N, -2, 1.5, -2, 2,
            maxIter
        )

        if img64 is not None:

            t64, _ = benchmark(
                mandelbrotOpenCL64,
                ctx, queue,
                N, -2, 1.5, -2, 2,
                maxIter
            )

            ratio = t64 / t32
            diff = np.abs(img32 - img64).max()

            print(f"OpenCL float64: {t64:.4f}s")
            print(f"Ratio: {ratio:.2f}x")
            print(f"Max diff: {diff}")

        else:
            t64 = None
            ratio = None
            diff = None

        results.append({
            "gridSize": N,
            "opencl32": t32,
            "opencl64": t64,
            "ratio": ratio,
            "diff": diff
        })

    return results

if __name__ == '__main__':
    main()