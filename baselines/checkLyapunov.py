import numpy as np
import scipy
from dynamics import A, B, kT, f, g
from controller import K

Acl = A-B.dot(K)
P = scipy.linalg.solve_continuous_lyapunov(Acl.T, -np.eye(A.shape[0]))

def V(x):
	x = x.reshape(-1,1)
	return x.T.dot(P).dot(x)

def DVDx(x):
	x = x.reshape(-1,1)
	return 2 * x.T.dot(P)

def dVdt(x):
	x = x.reshape(-1,1)
	return DVDx(x).dot(f(x, -K.dot(x) + np.array([0, 0, g / kT]).reshape(3, 1)).reshape(-1,1))

x_lb = np.array([-1, -1, -10 / 180. * np.pi, -1, -1, -1, -10 / 180. * np.pi, -1, -1, -1])
x_ub = np.array([1, 1, 10 / 180. * np.pi, 1, 1, 1, 10 / 180. * np.pi, 1, 1, 1])

LHS = []
RHS = []

for _ in range(1000):
	x = x_lb + np.random.rand(*x_lb.shape) * (x_ub - x_lb)
	LHS.append(dVdt(x).item())
	RHS.append(V(x).item())

LHS = np.array(LHS)
RHS = np.array(RHS)

_lambda = np.max(LHS / RHS)

from controller import cl_nonlinear
from scipy.integrate import odeint
dt = 0.01
x = x_lb + np.random.rand(*x_lb.shape) * (x_ub - x_lb)

# simulate
def simulate(x):
    return odeint(cl_nonlinear, x, [0, dt], args=(np.zeros(10),))[-1]

Vs = [V(x).item(), ]
ts = [0., ]
refs = [Vs[0], ]
for i in range(1000):
    t = i * dt
    x = simulate(x)
    Vs.append(V(x).item())
    ts.append(t)
    refs.append(np.exp(_lambda * t) * Vs[0])
from matplotlib import pyplot as plt
plt.plot(ts, Vs, label='sample')
plt.plot(ts, refs, label=r'$V(t) = e^{%.3f t} V(0)$'%_lambda)
plt.xlabel('t (s)')
plt.ylabel('V(x(t))')
plt.yscale('log')
plt.legend()
plt.show()