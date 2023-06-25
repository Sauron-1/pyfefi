import numpy as np
from pyfefi.line_tracer import LineTracer2
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 1000)
y = np.linspace(0, 2*np.pi, 1200)
X, Y = np.meshgrid(x, y, indexing='ij')

delta = np.array([x[1]-x[0], y[1] - y[0]])

u = np.sin(X - Y) * np.cos(X + Y) + np.sqrt(Y + X) * 5
Vx = -np.gradient(u, axis=1) / delta[1]
Vy = np.gradient(u, axis=0) / delta[0]
print(np.max(Vx))
#x = np.arange(len(x))
#X, Y = np.meshgrid(x, x, indexing='ij')

init = np.array([4*np.pi/3.1, 4*np.pi/3])
lt = LineTracer2(Vx, Vy, delta)
#l = lt.trace(np.array([400, 200]), 0.01, 1e-7, 1e-7, 1e4, 1e-2)
#l = lt.trace(np.array([400, 200]), min_step=1e-10)
#l = lt.trace(np.array([np.pi, np.pi]), min_move=0, min_step=1e-10)
l = lt.trace(init, max_step=1e-1, term_val=1e-5, tol_rel=1e-5)
print(len(l))

plt.pcolormesh(X, Y, u)
plt.streamplot(x, y, Vx.T, Vy.T)
plt.plot(l[:, 0], l[:, 1])
plt.scatter(init[0], init[1])
plt.xlim(x[0], x[-1])
plt.ylim(y[0], y[-1])
plt.show()
