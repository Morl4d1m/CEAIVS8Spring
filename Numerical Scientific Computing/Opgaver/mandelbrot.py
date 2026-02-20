import numpy as np
import matplotlib.pyplot as plt
import time
import statistics

def main():
    #mandelBrot(-2,1.5,-2,2,1024)
    medianSlow, resultSlow = benchmark(mandelBrot,-2,1.5,-2,2,1024)
    medianFast, resultFast = benchmark(mandelBrotFast,-2,1.5,-2,2,1024)
    if np.allclose(resultSlow,resultFast):
        print("Results match!")
    else:
        print("Results differ!")
    # Check where they differ :
    diff = np.abs(resultSlow - resultFast)
    print(f"Max difference: {diff.max ()}")
    print(f"Different pixels: {(diff>0).sum()}")
    
    # Speedup
    speedup = medianSlow / medianFast
    print(f"Speedup (medianSlow / medianFast): {speedup:.2f}x")


def mandelBrot(xMin, xMax, yMin, yMax, xyValues):
    # setting parameters (these values can be changed)
    xDomain, yDomain = np.linspace(xMin,xMax,xyValues), np.linspace(yMin,yMax,xyValues)
    bound = 2
    power = 2             # any positive floating point value (n)
    max_iterations = 50   # any positive integer value
    colormap = 'magma'    # set to any matplotlib valid colormap


    # computing 2-d array to represent the mandelbrot-set
    iterationArray = []
    for y in yDomain:
        row = []
        for x in xDomain:
            c = complex(x,y)
            z = 0
            for iterationNumber in range(max_iterations):
                if(abs(z) >= bound):
                    row.append(iterationNumber)
                    break
                else: z = z**power + c
            else:
                row.append(0)

        iterationArray.append(row)

    # plotting the data
    ax = plt.axes()
    #plt.rc('text', usetex = True)   # adding this line so that tex can be used
    ax.set_aspect('equal')
    graph = ax.pcolormesh(xDomain, yDomain, iterationArray, cmap = colormap)
    plt.colorbar(graph)
    plt.xlabel("Real-Axis")
    plt.ylabel("Imaginary-Axis")
    plt.title('Multibrot set for $z_{{new}} = z^{{{}}} + c$'.format(power))
    plt.gcf().set_size_inches(5,4)
    #plt.show()
    return np.array(iterationArray)


def mandelBrotFast(xMin, xMax, yMin, yMax, xyValues):
    # setting parameters (these values can be changed)
    xDomain, yDomain = np.linspace(xMin,xMax,xyValues), np.linspace(yMin,yMax,xyValues)
    bound = 2
    power = 2             # any positive floating point value (n)
    max_iterations = 50   # any positive integer value
    colormap = 'magma'    # set to any matplotlib valid colormap


    # computing 2-d array to represent the mandelbrot-set
    X, Y = np.meshgrid(xDomain,yDomain)
    C = X + 1j*Y
    #print(f"Shape: {C.shape}")
    #print(f"Type: {C.dtype}")
    # Z starts at 0 everywhere, M counts iterations
    Z = np.zeros_like(C, dtype=np.complex128)
    M = np.zeros(C.shape, dtype=np.int32)

    # Mandelbrot iterations (only loop we keep)
    for _ in range(max_iterations):
        mask = np.abs(Z) <= bound  # points not escaped yet

        # Update Z only where not escaped
        Z[mask] = Z[mask]**power + C[mask]

        # Count iterations only where not escaped
        M[mask] += 1

    # plotting the data
    ax = plt.axes()
    #plt.rc('text', usetex = True)   # adding this line so that tex can be used
    ax.set_aspect('equal')
    graph = ax.pcolormesh(xDomain, yDomain, M, cmap = colormap)
    plt.colorbar(graph)
    plt.xlabel("Real-Axis")
    plt.ylabel("Imaginary-Axis")
    plt.title('Multibrot set for $z_{{new}} = z^{{{}}} + c$'.format(power))
    plt.gcf().set_size_inches(5,4)
    #plt.show()
    M[M == max_iterations] = 0

    return M


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

if __name__ == '__main__':
    main()