__author__ = 'koorosh'
import sympy as sym
from sympy.physics.quantum import Operator

k1, k3, m, g = sym.symbols('k1 k3 m g', real=True)
D0, D1, D2 = sym.symbols('D0 D1 D2', real=True)
eps = sym.symbols('eps', real=True)
omega0, alpha2 = sym.symbols('omega0, alpha2', real=True)
x0, x1, x2, x3 = sym.symbols('x0 x1 x2 x3', real=True)
T0, T1, T2 = sym.symbols('T0 T1 T2', real=True)
A, A1, A2 = sym.symbols('A A1 A2', real=True)
a = sym.symbols('a', real=True)
D0 = Operator('D0')
D1 = Operator('D1')
# # # First we solve for the stationary point of k3 * x**3 + k1 * x - mg = 0
# # # See this link for more details: http://goo.gl/4fAj7K
# A = k1 / k3
# B = m * g / k3
# u1 = (-B + sym.sqrt(B**2 + 4 * A**3 / 27)) / 2
# u2 = (-B - sym.sqrt(B**2 + 4 * A**3 / 27)) / 2
# t1 = u1**(1/3)
# t2 = u2**(1/3)
# s1 = A / (3 * t1)
# s2 = A / (3 * t2)
# x01 = s1 - t1 # First stationary point
# x02 = s2 - t2 # Second stationary point

ddt2 = D0**2 + 2 * eps * D0 * D1 + eps**2 * (D1**2 + 2 * D0 * D2)
ddt = D0 + eps * D1
x = x0 + eps * x1

equ = ddt2 * x + omega0**2 * x + eps * alpha2 * x**3 - g
equ = equ.expand()

print(sym.pretty(equ.subs({eps:0})))
print(sym.pretty(equ.coeff(eps)))
print(sym.pretty(equ.coeff(eps**2)))
# print(sym.pretty(equ.coeff(eps**3)))



# A1 = a * sym.exp(-3 * alpha2 / (2 * omega0**5) * T1)
# # Equation for eps0
X0 = A(T1) * sym.exp(sym.I * omega0 * T0) + g / omega0**2
# X0 = A1(T1) * A2(T2) * sym.exp(sym.I * omega0 * T0) + g / omega0**2
# X0 = A1 * A2(T2) * sym.exp(sym.I * omega0 * T0) + g / omega0**2
# equEps0 = sym.diff(sym.diff(X0, T0), T0) - g + omega0**2 * X0
# equEps0 = equEps0.expand()
# print(sym.pretty(equEps0))

# # Equation for eps1
# equEps1RHS = -(2 * sym.diff(sym.diff(X0, T1), T0) +
#                alpha2 * X0**3)
# equEps1RHS = sym.simplify(equEps1RHS.expand())
# print(sym.pretty(equEps1RHS))
# print(sym.pretty(equEps1RHS.coeff(sym.exp(sym.I * omega0 * T0))))
#
# fid = open('equation.txt', 'w')
# fid.write(sym.latex(equEps1RHS, mode='equation'))
# fid.close()

# A1 = a * sym.exp(-3 * alpha2 / (2 * omega0**5) * T1)
# X1 = -alpha2 * g**3 / omega0**8 - \
#      3 * alpha2 * g * A1**2 * A2**2 / (omega0**2 * (omega0**2 - 4 * omega0**2)) * \
#         sym.exp(2 * sym.I * T0 * omega0) - \
#      alpha2 * A1**3 + A2**3 / (omega0**2 - 9 * omega0**2) * \
#         sym.exp(3 * sym.I * T0 * omega0)
#
# # # Equation for eps2
# equEps2RHS = -(2 * sym.diff(sym.diff(X1, T1), T0) +
#                2 * sym.diff(sym.diff(X0, T2), T0) +
#                1 * sym.diff(sym.diff(X0, T1), T1) +
#                3 * alpha2 * X0**2 * X1)
# equEps2RHS = sym.simplify(equEps2RHS.expand())
# # print(sym.pretty(equEps2RHS))
# # print(sym.pretty(equEps2RHS.coeff(sym.exp(sym.I * omega0 * T0))))
