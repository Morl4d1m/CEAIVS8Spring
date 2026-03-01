import numpy as np
import matplotlib.pyplot as plt
import time
import statistics
import numba
from numba import jit, njit
import cProfile, pstats

cProfiling = 0
lineProfiling = 0
makePlots = 1
gridPlot = 1


def main():
    #mandelBrot(-2,1.5,-2,2,1024)
    print("Program begun")
    
    if makePlots and gridPlot:
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for row, dType in enumerate([np.float16, np.float32, np.float64]): # For fun add: , np.int8, np.int16, np.int32]:
        mandelBrotNumba(-2,1.5,-2,2,64,dType) # Warmup
        print(f"Numba warmed up using {dType.__name__}.")
        print(f"Benchmarking begins for {dType.__name__}:")
        medianSlow, resultSlow = benchmark(mandelBrot,-2,1.5,-2,2,1024,dType)
        medianFast, resultFast = benchmark(mandelBrotFast,-2,1.5,-2,2,1024,dType)
        medianNumba, resultNumba = benchmark(mandelBrotNumba,-2,1.5,-2,2,1024,dType)
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
        else:
            print("No plots made")

    if cProfiling: 
        print("cProfile testing initiated:")
        cProfile.run("mandelBrot(-2,1.5,-2,2,1024,np.float64)",'naiveProfile.prof')
        cProfile.run("mandelBrotFast(-2,1.5,-2,2,1024,np.float64)",'fastProfile.prof')
        for name in ('naiveProfile.prof', 'fastProfile.prof'):
            stats = pstats.Stats(name)
            stats.sort_stats('cumulative')
            stats.print_stats(10)
    else:
        print("cProfiling has not been done this run")

    if makePlots:
        if gridPlot:
            plt.tight_layout()
        plt.show()



def mandelBrot(xMin, xMax, yMin, yMax, xyValues,dataType):
    xDomain, yDomain = np.linspace(xMin,xMax,xyValues).astype(dataType), np.linspace(yMin,yMax,xyValues).astype(dataType)
    bound = 2
    power = 2             # any positive floating point value (n)
    maxIterations = 50   # any positive integer value
    #colormap = 'magma'    # set to any matplotlib valid colormap

    # computing 2-d array to represent the mandelbrot-set
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
    """ # Replaced by plotting function
    # plotting the data
    ax = plt.axes()
    ax.set_aspect('equal')
    graph = ax.pcolormesh(xDomain, yDomain, iterationArray, cmap = colormap)
    plt.colorbar(graph)
    plt.xlabel("Real-Axis")
    plt.ylabel("Imaginary-Axis")
    plt.title('Multibrot set for $z_{{new}} = z^{{{}}} + c$'.format(power))
    plt.gcf().set_size_inches(5,4)
    #plt.show()"""
    return np.array(iterationArray)


def mandelBrotFast(xMin, xMax, yMin, yMax, xyValues,dataType):
    xDomain, yDomain = np.linspace(xMin,xMax,xyValues).astype(dataType), np.linspace(yMin,yMax,xyValues).astype(dataType)
    bound = 2
    power = 2             # any positive floating point value (n)
    maxIterations = 50   # any positive integer value
    #colormap = 'magma'    # set to any matplotlib valid colormap


    # computing 2-d array to represent the mandelbrot-set
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
    """ # Replaced by plotting function
    # plotting the data
    ax = plt.axes()
    ax.set_aspect('equal')
    graph = ax.pcolormesh(xDomain, yDomain, M, cmap = colormap)
    plt.colorbar(graph)
    plt.xlabel("Real-Axis")
    plt.ylabel("Imaginary-Axis")
    plt.title('Multibrot set for $z_{{new}} = z^{{{}}} + c$'.format(power))
    plt.gcf().set_size_inches(5,4)
    #plt.show()"""
    M[M == maxIterations] = 0

    return M

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


def benchmark(func, *args, nRuns=10):
    """Time func, return median of nRuns"""
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
    mesh = ax.pcolormesh(xDomain, yDomain, result,
                         cmap="magma", shading="auto")
    ax.set_xlabel("Real-Axis")
    ax.set_ylabel("Imaginary-Axis")
    ax.set_title(title)
    ax.set_aspect("equal")
    return mesh


if __name__ == '__main__':
    main()