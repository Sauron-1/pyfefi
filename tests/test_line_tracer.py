import numpy as np
from pyfefi.line_tracer import LineTracer2
import matplotlib.pyplot as plt

test_one = False

x = np.linspace(0, 2*np.pi, 1000, dtype=np.float32)
y = np.linspace(0, 2*np.pi, 1200, dtype=np.float32)
#x = np.linspace(0, 2*np.pi, 1000)
#y = np.linspace(0, 2*np.pi, 1200)
X, Y = np.meshgrid(x, y, indexing='ij')

delta = np.array([x[1]-x[0], y[1] - y[0]])

u = np.sin(X - Y) * np.cos(X + Y) + np.sqrt(Y + X) * 5
Vx = -np.gradient(u, axis=1) / delta[1]
Vy = np.gradient(u, axis=0) / delta[0]
print(np.max(Vx))
#x = np.arange(len(x))
#X, Y = np.meshgrid(x, x, indexing='ij')

delta = delta.astype(np.float64)
print('Building line tracer...', end='')
lt = LineTracer2(Vx, Vy, delta, np.zeros(2))
print('Done')

plt.pcolormesh(X, Y, u)
#plt.streamplot(x, y, Vx.T, Vy.T)

if test_one:
    init = np.array([4*np.pi/3.1, 4*np.pi/3])
    l = lt.trace(init, max_step=1e-2, term_val=0, tol_rel=1e-7)
    print(len(l))

    plt.plot(l[:, 0], l[:, 1], color='r')
    plt.scatter(init[0], init[1], color='r')
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.show()

else:
    #inits = np.array([
    #    [3*np.pi/3.1, 3*np.pi/3],
    #    [4*np.pi/3.1, 4*np.pi/3],
    #    [5*np.pi/3.1, 5*np.pi/3],
    #])
    inits = []
    for factor in np.linspace(0, 6, 100):
        inits.append([factor*np.pi/3, factor*np.pi/3])
    inits = np.array(inits)

    for i, init in enumerate(inits):
        l0 = lt.trace(init, max_step=100, term_val=1e-9, tol_rel=1e-4, tol=1e-4, max_iter=500)
    print('Starting trace')
    ls = lt.trace_many(inits, max_step=100, term_val=1e-9, tol_rel=1e-4, tol=1e-4, max_iter=500)
    print('Trace done')
    #roots = lt.find_roots(inits, max_step=1e-1, term_val=1e-5, tol_rel=1e-5)
    #print(roots)
    for i in range(len(inits)):
        l = ls[i]
        init = inits[i]
        plt.plot(l[:, 0], l[:, 1], color='r')
        plt.scatter(init[0], init[1], color='r')

    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.savefig('test.png')
    plt.show()
