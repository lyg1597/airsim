### From https://arxiv.org/pdf/1703.07373.pdf Eq. (19) (Left)
import numpy as np
from numpy import cos, sin

# quadrotor physical constants
g = 9.81 
R = 0.1 
l = 0.5
Mr = 0.1 
M = 1 
m = M+4*Mr

Jx = 2/5*M*R**2+2*l**2*Mr
Jy = Jx 
Jz = 2/5*M*R**2+4*l**2*Mr

# non-linear dynamics
def f(x, u):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = x.reshape(-1).tolist()
    F, t_phi, t_theta, t_psi = u.reshape(-1).tolist()
    dot_x = np.array([
        cos(x9)*x4-sin(x9)*x5,
        sin(x9*x4)+cos(x9)*x5,
        -x6,
        x12*x5-x11*x6-g*sin(x8),
        x10*x6-x12*x4+g*cos(x8)*sin(x7),
        x11*x4-x10*x5+g*cos(x8)*cos(x7)-F/m,
        x10,
        x11,
        x12,
        (Jy-Jz)/Jx*x11*x12+1/Jx*t_phi,
        (Jz-Jx)/Jy*x10*x12+1/Jy*t_theta,
        (Jx-Jy)/Jz*x10*x11+1/Jz*t_psi 
    ])
    return dot_x

# linearization
# The state variables are x, y, z, vx, vy, vz, theta_x, theta_y, omega_x, omega_y
A = np.zeros([12, 12])
A[0,3] = 1. 
A[1,4] = 1.
A[2,5] = -1.
A[3,7] = -g
A[4,6] = g
A[5,6] = g 
A[5,7] = g 
A[6, 9] = 1 
A[7,10] = 1 
A[8,11] = 1 

# A[0, 1] = 1.
# A[1, 2] = g
# A[2, 2] = -d1
# A[2, 3] = 1
# A[3, 2] = -d0
# A[4, 5] = 1.
# A[5, 6] = g
# A[6, 6] = -d1
# A[6, 7] = 1
# A[7, 6] = -d0
# A[8, 9] = 1.

B = np.zeros([12, 4])
B[5,0] = -1/m 
B[9,1] = 1/Jx
B[10,2] = 1/Jy
B[11,3] = 1/Jz
