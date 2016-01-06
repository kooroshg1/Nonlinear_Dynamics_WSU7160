import sympy as sym
import numpy as np
import mpmath as mp
from scipy.linalg import eig
from scipy.linalg import eigh
import pdb

'''
The Following calculates the Jacobian of the State function F in the symbolic form. It also calcualtes the eigenvalue and eigenvector of the Jacobian matrix. If required, it will substitude the value for the stationary point inside the Jacobian inorder to do the stability analysis.
'''

# Example 1 - Pendulum
# m \frac{d^2 \theta}{d t^2} + \mu \frac{d\theta}{dt} + k sin(\theta)
# Define state space equations
# \dot{x}_1 = x_2
# \dot{x}_2 = -\frac{\mu}{m} x_2 - \frac{k}{m} sin(x_1)

# Define variables in symbolic form
x1, x2, mu, k, m = sym.symbols('x1 x2 mu k m')
# Define the state matrix
F = sym.Matrix([x2, -mu/m*x2 - k/m*sym.sin(x1)])
# Calculate the Jacobian of the state matrix
JacF = F.jacobian([x1, x2])
# Evaluate the Jacobian at the stationary point
JacF = JacF.subs([(x1, 2*sym.pi), (x2, 0), (m, 1.0), (k, 1.0), (mu, 0.1)]).evalf()
print(JacF)
print('Symbolic Eigenvalues: ')
print(sym.pretty(JacF.eigenvects()))
print()

A = sym.Matrix([[7, 1],[-4, 3]])
print((sym.pretty(A.eigenvects())))

JacF = np.array(JacF)
eval , evec = eig(A)
print('Eigenvectors: ')
print(evec)
print('Eigenvalues: ')
print(eval)

'''
# Example 2 - Pendulum
# van der pol system
# Define state space equations
# \dot{u}_1 = v
# \dot{v}_2 = -(u^2 - 1)v - u

# Define variables in symbolic form
u, v = sym.symbols('u v')
# Define the state matrix
F = sym.Matrix([v, -(u**2 - 1)*v - u])
# Calculate the Jacobian of the state matrix
JacF = F.jacobian([u, v])
# Evaluate the Jacobian at the stationary point
JacF = JacF.subs([(u, 0), (v, 0)]).evalf()
print(JacF)
print('Symbolic Eigenvalues: ')
print(JacF.eigenvals())
print()

JacF = np.array(JacF)
eval , evec = eig(JacF)
print('Eigenvectors: ')
print(evec)
print('Eigenvalues: ')
print(eval)
'''
