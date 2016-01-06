__author__ = 'koorosh'
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

alpha = 12
Tf = 2 * np.pi
# State Matrix
def RHS(X, t=0.0):
    x1, x2 = X
    x1dot = x2
    x2dot = -(0.01 + alpha / 100) * x2 - x1 - alpha / 10 * x1**3 + np.sin(3*t)
    return [x1dot, x2dot]

t = np.linspace(0.0, Tf, 5000)
sol = odeint(RHS, [0, 0], t)
x = sol[:, 0]

# State Matrix
tFloquet = np.linspace(0.0, 2*np.pi, 100)
def A(Y, t=0.0):
    y1, y2, y3 = Y
    x10 = 0.12494 * np.sin(3 * t)
    y1dot = y2
    y2dot = -(0.01 + alpha / 100) * y2 - (1 + 0.3 * alpha * x10**2) * y1
    y3dot = 0
    return [y1dot, y2dot, y3dot]

sol1 = odeint(A, [1, 0, 0], tFloquet)
sol2 = odeint(A, [0, 1, 0], tFloquet)
sol3 = odeint(A, [0, 0, 1], tFloquet)
# sol1 = odeint(A, [1, 0, 0], t)
# sol2 = odeint(A, [0, 1, 0], t)
# sol3 = odeint(A, [0, 0, 1], t)

PHI = np.array([sol1[-1, :], sol2[-1, :], sol3[-1, :]])
# PHI = np.array([sol1[-1, :], sol2[-1, :]])
eigenValue = np.linalg.eigvals(PHI)
print(eigenValue)
print(np.abs(eigenValue))