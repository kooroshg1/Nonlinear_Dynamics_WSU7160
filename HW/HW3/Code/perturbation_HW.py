import sympy as sym
from sympy.physics.quantum import Operator
from sympy.physics.quantum.operator import DifferentialOperator

D0, D1, D2 = sym.symbols('D0 D1 D2', real=True)
x, x0, x1, x2, x3 = sym.symbols('x x0 x1 x2 x3', real=True)
eps, omega0, mu = sym.symbols('eps omega0 mu', real=True)
A, B = sym.symbols('A B', real=True)
T0, T1, T2 = sym.symbols('T0 T1 T2', real=True)
D0 = Operator('D0')
D1 = Operator('D1')
f = sym.Function('f')
d = DifferentialOperator(sym.diff(f(x), x), f(x))
print(sym.pretty(d(1/x)))
# ddt2 = D0**2 + 2 * eps * D0 * D1 + eps**2 * (D1**2 + 2 * D0 * D2)
# ddt = D0 + eps * D1

ddt2 = D0**2 + 2 * eps * D0 * D1
ddt = D0 + eps * D1

# x = x0 + eps * x1 + eps**2 * x2 + eps**3 * x3
x = x0 + eps * x1

f = ddt * x - (ddt * x)**3
equ = ddt2 * x + omega0**2 * x - eps * f
# equ = ddt2 * x + 2 * eps * mu * ddt * x + x + eps * x**3
equ = equ.expand()

print(sym.pretty(equ.subs({eps:0})))
print(sym.pretty(equ.coeff(eps)))
# print(sym.pretty(equ.coeff(eps**2)))
# print(sym.pretty(equ.coeff(eps**3)))

X0 = A(T1) * sym.exp(sym.I * T0)# + sym.conjugate(A(T1) * sym.exp(sym.I * T0))
# X1 = A(T2) * sym.exp(T1 / 2) * sym.exp(sym.I * omega0 * T0) + sym.conjugate(A(T2) * sym.exp(T1 / 2) * sym.exp(sym.I * omega0 * T0))
# # Use below to check if above satisfies the first equation
# equEps = (sym.diff(sym.diff(X1, T0), T0) + omega0**2 * X1).expand()
equEps = (D0**2 * X0 + omega0**2 * X0).expand()
print(sym.pretty(equEps))

# X2 = B(T1, T2) * sym.exp(sym.I * omega0 * T0) + sym.conjugate(B(T1, T2) * sym.exp(sym.I * omega0 * T0))
# Use below to check that above results in no secular terms on the right-hand-side of eps**2 equation
# equEps1RHS = -(1 * sym.diff(sym.diff(sym.diff(X0, T0), T0), T0) +
#                2 * sym.diff(sym.diff(X0, T0), T1) -
#                1 * sym.diff(X0, T0)).expand()
# print(sym.pretty(equEps1RHS))

