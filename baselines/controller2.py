import dynamics2
from dynamics2 import g, A, B
import numpy as np
import scipy
from scipy.integrate import odeint
import copy
import matplotlib.pyplot as plt 

def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.
    dx/dt = A x + B u
    cost = integral x.T*Q*x + u.T*R*u
    """
    # http://www.mwm.im/lqr-controllers-with-python/
    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R) * (B.T * X))

    eigVals, eigVecs = scipy.linalg.eig(A - B * K)

    return np.asarray(K), np.asarray(X), np.asarray(eigVals)

####################### solve LQR #######################
print(A)
print(B)
n = A.shape[0]
m = B.shape[1]
Q = np.eye(n)
Q[0, 0] = 20.
Q[1, 1] = 20.
# Q[2, 2] = 10.
# Q[5, 5] = 10.
R = np.diag([1., 1., 1., 1.])
K, _, _ = lqr(A, B, Q, R)
print(K)

####################### The controller ######################
def u(x, goal):
    goal = np.array(goal)
    return K.dot([
        goal[0],goal[1],goal[2],
        0,0,goal[3],0,0,0,
        0,0,0
    ] - x)

######################## The closed_loop system #######################
def cl_nonlinear(x, t, goal):
    x = np.array(x)
    dot_x = dynamics2.f(x, u(x, goal))
    return dot_x

# simulate
def simulate(x, goal, dt):
    curr_position = np.array(x)[[0,1,2]]
    goal_position = np.array(goal)[[0,1,2]]
    error = goal_position - curr_position
    distance = np.sqrt((error**2).sum())
    if distance > 1:
        goal[0:3] = curr_position + error / distance
    error_angle = x[5] - goal[3]
    if abs(error_angle) > np.deg2rad(10):
        goal[3] = x[5] + error_angle/abs(error_angle)*np.deg2rad(10)
    return odeint(cl_nonlinear, x, [0, dt], args=(goal,))[-1]

if __name__ == "__main__":
    x0 = np.zeros(12)
    dt = 0.01 
    goal = [5,5,5,0]
    x_list = [copy.deepcopy(x0)]
    for i in range(1000):
        res = simulate(x0, goal, dt)
        x_list.append(copy.deepcopy(res))
        x0 = res
    x_list = np.array(x_list)
    print(x_list.shape)
    plt.figure(0)
    plt.plot(x_list[:,0])
    plt.figure(1)
    plt.plot(x_list[:,1])
    plt.figure(2)
    plt.plot(x_list[:,2])
    plt.figure(3)
    plt.plot(x_list[:,5])
    plt.show()