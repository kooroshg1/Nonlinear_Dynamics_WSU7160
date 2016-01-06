import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys
import pdb

linewidth = 3.0
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 62}
plt.rc('font', **font)

# User input
epsUsr = -0.1
kUsr = 1.0
m1Usr = 1.0
gUsr = 1.0
lUsr = 1.0
nInitial = 10 # Number of trajectories
nPoints = 40 # Number of points for making the phase portrait (quiver plot)
tf = 5.0 # Final time for trajectory calculation (final time in odeint time integration)
nt = 101 # Number of time steps in odeint
case_numer = 1
scaleArrow = 6.0
copyToDist = False

# --------------------------------------------------------------------
# --------------------------------------------------------------------
''' The Following is the numerical solution of the nonlinear problem '''
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Define F (state function!)
def F(x1_x2, t=0.0, eps=0.0, k=1.0, m1=1.0, g=1.0, l=1.0):
    x1, x2 = x1_x2
    x1dot = x2
    x2dot = -(eps * x1 * x2**2.0 / (1 - x1**2.0)**2.0 +
              k * x1 / m1 +
              eps * g * x1 / (l * np.sqrt(1.0 - x1**2.0))) / \
        (1.0 + eps * x1**2.0 / (1.0 - x1**2.0))
    return [x1dot, x2dot]

# Plot the phase portrait
x1 = np.linspace(-1.0 + sys.float_info.epsilon, 1.0 - sys.float_info.epsilon, nPoints)
x2 = np.linspace(-1.0 + sys.float_info.epsilon, 1.0 - sys.float_info.epsilon, nPoints)
X1, X2 = np.meshgrid(x1, x2)
X1dot = np.zeros(X1.shape)
X2dot = np.zeros(X2.shape)

# Calculating the slope of arrows for the phase portrait
plt.figure(figsize=(20, 20))
for ix in range(0, len(x1)):
    for jy in range(0, len(x2)):
        X1dot_, X2dot_ = F([X1[ix, jy], X2[ix, jy]], t=0.0, eps=epsUsr, k=kUsr, m1=m1Usr, g=gUsr, l=lUsr)
        X1dot[ix, jy] = X1dot_ / np.sqrt(X1dot_**2.0 + X2dot_**2.0)
        X2dot[ix, jy] = X2dot_ / np.sqrt(X1dot_**2.0 + X2dot_**2.0)


# Calculating the trajectories for multiple boundary conditions
X10 = 2.0 * np.random.rand(nInitial, 1) - 1.0
X20 = 2.0 * np.random.rand(nInitial, 1) - 1.0
# x10 = np.linspace(-1.0 + sys.float_info.epsilon, 1.0 - sys.float_info.epsilon, nInitial)
# x20 = np.linspace(-1.0 + sys.float_info.epsilon, 1.0 - sys.float_info.epsilon, nInitial)
# X10, X20 = np.meshgrid(x10, x20)
# X10 = X10.flatten()
# X20 = X20.flatten()
t = np.linspace(0.0, tf, nt)
for i0 in range(0, len(X10)):
    init = [X10[i0, 0], X20[i0, 0]]
    # init = [X10[i0], X20[i0]]
    SOL = odeint(F, init, t, (epsUsr, kUsr, m1Usr, gUsr, lUsr))
    nSOL = len(t)
    for iSOL in range(0, nSOL):
        if SOL[iSOL, 0] >= 1.0 or SOL[iSOL, 1] > 2.0 or SOL[iSOL, 0] <= -1.0 or SOL[iSOL, 1] < -2.0:
            nSOL = iSOL - 1
            break
        if np.isnan(SOL[iSOL, 0]) or np.isnan(SOL[iSOL, 0]):
            nSOL = iSOL - 1
            break
    # pdb.set_trace()
    plt.plot(SOL[0:nSOL, 0], SOL[0:nSOL, 1], 'b',
             SOL[0, 0], SOL[0, 1], 'r>',
             SOL[nSOL-1, 0], SOL[nSOL-1, 1], 'rs', ms=15, lw=linewidth)
    # arrowLoc = round(nSOL/2)
    # plt.arrow(SOL[arrowLoc, 0], SOL[arrowLoc, 1],
    #           0.01 * (SOL[arrowLoc + 1, 0] - SOL[arrowLoc, 0]), 0.01 * (SOL[arrowLoc + 1, 1] - SOL[arrowLoc, 1]),
    #           color='r', head_width=0.03, head_length=0.06)

plt.quiver(X1, X2, X1dot, X2dot)
plt.xlim([x1[0], x1[-1]])
plt.ylim([x2[0], x2[-1]])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
if copyToDist:
        filename = str('/home/koorosh/Desktop/ME7160/Exam/Midterm_1_Takehome/report/figure/' +
                       'phase_portrait_l2nlcomp' + str(int(
                case_numer)) + '.eps')
else:
        filename = str('phase_portrait_' + str(int(case_numer)) + '.eps')
plt.savefig(filename, bbox_inches='tight')
plt.show()

# --------------------------------------------------------------------
# --------------------------------------------------------------------
''' The Following is symbolic operation '''
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Taylor series expansion example for sin(x) around x0 = pi/2
# x = sym.symbols('x')
# print(sym.pretty(sym.series(sym.sin(x), x, x0=sym.pi / 2)))

eps, k, m1, g, l = sym.symbols('eps k m1 g l')
x1, x2, x1dot, x2dot = sym.symbols('x1 x2 x1dot x2dot')
y1dot, y2dot, y1, y2 = sym.symbols('y1dot y2dot y1 y2')

# Define f1 and f2 in symbolic way for differentiation
f1 = x2
f2 = -(eps * x1 * x2**2.0 / (1.0 - x1**2.0)**2.0 +
       k * x1 / m1 +
       eps * g * x1 / (l * sym.sqrt(1.0 - x1**2.0))) / \
     (1.0 + eps * x1**2.0 / (1.0 - x1**2.0))

# First derivative of f1 to x1 and x2
df1_dx1 = sym.diff(f1, x1)
df1_dx2 = sym.diff(f1, x2)
# Second derivative of f1 to x1 and x2
ddf1_ddx1 = sym.diff(df1_dx1, x1)
ddf1_dx1dx2 = sym.diff(df1_dx1, x2)
ddf1_ddx2 = sym.diff(df1_dx2, x2)
# Third derivative of f1 to x1 and x2
dddf1_dddx1 = sym.diff(ddf1_ddx1, x1)
dddf1_ddx1dx2 = sym.diff(ddf1_dx1dx2, x1)
dddf1_dx1ddx2 = sym.diff(ddf1_dx1dx2, x2)
dddf1_dddx2 = sym.diff(ddf1_ddx2, x2)

# First derivative of f2 to x1 and x2
df2_dx1 = sym.diff(f2, x1)
df2_dx2 = sym.diff(f2, x2)
# Second derivative of f2 to x1 and x2
ddf2_ddx1 = sym.diff(df2_dx1, x1)
ddf2_dx1dx2 = sym.diff(df2_dx1, x2)
ddf2_ddx2 = sym.diff(df2_dx2, x2)
# Third derivative of f2 to x1 and x2
dddf2_dddx1 = sym.diff(ddf2_ddx1, x1)
dddf2_ddx1dx2 = sym.diff(ddf2_dx1dx2, x1)
dddf2_dx1ddx2 = sym.diff(ddf2_dx1dx2, x2)
dddf2_dddx2 = sym.diff(ddf2_ddx2, x2)
# Linearizing x1dot and x2dot around x10 and x20
# First stationary point
# x10 = 0.0
# x20 = 0.0
# Second stationary point
# x10 = sym.sqrt(1.0 - (m1 * eps * g / (k * l))**2.0)
# x20 = 0.0
# Third stationary point
x10 = -sym.sqrt(1.0 - (m1 * eps * g / (k * l))**2.0)
x20 = 0.0

y1dot = f1 + \
    df1_dx1 * y1 + df1_dx2 * y2 + \
    1/2.0 * (ddf1_ddx1 * y1**2.0 + 2.0 * ddf1_dx1dx2 * y1 * y2 + ddf1_ddx2 * y2**2.0) + \
    1/6.0 * (dddf1_dddx1 * y1**3.0 + 3.0 * dddf1_ddx1dx2 * y1**2.0 * y2 + 3.0 * dddf1_dx1ddx2 * y1 * y2**2.0 +
             dddf1_dddx2 * y2**3.0)

y2dot = f2 + \
    df2_dx1 * y1 + df2_dx2 * y2 + \
    1/2.0 * (ddf2_ddx1 * y1**2.0 + 2.0 * ddf2_dx1dx2 * y1 * y2 + ddf2_ddx2 * y2**2.0) + \
    1/6.0 * (dddf2_dddx1 * y1**3.0 + 3.0 * dddf2_ddx1dx2 * y1**2.0 * y2 + 3.0 * dddf2_dx1ddx2 * y1 * y2**2.0 +
             dddf2_dddx2 * y2**3.0)

# print(sym.pretty(y1dot.subs([(x1, x10), (x2, x20)])))
# print()
# print(sym.pretty(y2dot.subs([(x1, x10), (x2, x20)])))
#
# fid = open('equation.txt', 'w')
# fid.write(sym.latex(y2dot.subs([(x1, x10), (x2, x20)]), mode='equation'))
# fid.close()

# Plot the expanded state space near the stationary point of interest
if type(x10) != float:
        x10 = float(x10.subs([(eps, epsUsr), (k, kUsr), (m1, m1Usr), (g, gUsr), (l, lUsr)]).evalf())
if type(x20) != float:
        x20 = float(x20.subs([(eps, epsUsr), (k, kUsr), (m1, m1Usr), (g, gUsr), (l, lUsr)]).evalf())

print([x10, x20])
X1 = np.linspace(-1.0, 1.0, nPoints)
X2 = np.linspace(-1.0, 1.0, nPoints)
X1, X2 = np.meshgrid(X1, X2)
X1dot = np.zeros(X1.shape)
X2dot = np.zeros(X2.shape)
# Calculating the slope of arrows for the phase portrait
for ix in range(0, nPoints):
    for jy in range(0, nPoints):
        X1dot_ = float(y1dot.subs([(x1, x10), (x2, x20), (y1, X1[ix, jy]), (y2, X2[ix, jy]), (eps, epsUsr), (k, kUsr), (m1, m1Usr), (g, gUsr), (l, lUsr)]).evalf())
        X2dot_ = float(y2dot.subs([(x1, x10), (x2, x20), (y1, X1[ix, jy]), (y2, X2[ix, jy]), (eps, epsUsr), (k, kUsr), (m1, m1Usr), (g, gUsr), (l, lUsr)]).evalf())
        X1dot[ix, jy] = X1dot_ / np.sqrt(X1dot_**2.0 + X2dot_**2.0)
        X2dot[ix, jy] = X2dot_ / np.sqrt(X1dot_**2.0 + X2dot_**2.0)

def Fl(y1_y2, t=0.0, epsUsr=0.0, kUsr=1.0, m1Usr=1.0, gUsr=1.0, lUsr=1.0):
    y1_, y2_ = y1_y2
    Y1DOT = y1dot.subs([(x1, x10), (x2, x20), (y1, y1_), (y2, y2_), (eps, epsUsr), (k, kUsr), (m1, m1Usr), (g, gUsr), (l, lUsr)]).evalf()
    Y2DOT = y2dot.subs([(x1, x10), (x2, x20), (y1, y1_), (y2, y2_), (eps, epsUsr), (k, kUsr), (m1, m1Usr), (g, gUsr), (l, lUsr)]).evalf()
    # Y1dot = Y1DOT.coeff('y1') * y1_ + Y1DOT.coeff('y2') * y2_
    # Y2dot = Y2DOT.coeff('y1') * y1_ + Y2DOT.coeff('y2') * y2_
    # pdb.set_trace()
    return [Y1DOT, Y2DOT]
# Calculating the trajectories for multiple boundary conditions
# X10 = 2.0 * np.random.rand(nInitial, 1) - 1.0
# X20 = 2.0 * np.random.rand(nInitial, 1) - 1.0
plt.figure(figsize=(20, 20))
# x10 = np.linspace(-1.0 + sys.float_info.epsilon, 1.0 - sys.float_info.epsilon, nInitial)
# x20 = np.linspace(-1.0 + sys.float_info.epsilon, 1.0 - sys.float_info.epsilon, nInitial)
# X10, X20 = np.meshgrid(x10, x20)
# X10 = X10.flatten()
# X20 = X20.flatten()
t = np.linspace(0.0, tf, nt)
for i0 in range(0, len(X10)):
    init = [X10[i0, 0], X20[i0, 0]]
    # init = [X10[i0], X20[i0]]
    SOL = odeint(Fl, init, t, args=(epsUsr, kUsr, m1Usr, gUsr, lUsr))
    nSOL = len(t)
    for iSOL in range(0, nSOL):
        if SOL[iSOL, 0] >= 1.0 or SOL[iSOL, 1] > 2.0 or SOL[iSOL, 0] <= -1.0 or SOL[iSOL, 1] < -2.0:
            nSOL = iSOL - 1
            break
        if np.isnan(SOL[iSOL, 0]) or np.isnan(SOL[iSOL, 0]):
            nSOL = iSOL - 1
            break
    plt.plot(SOL[0:nSOL, 0], SOL[0:nSOL, 1], 'b',
             SOL[0, 0], SOL[0, 1], 'r>',
             SOL[nSOL-1, 0], SOL[nSOL-1, 1], 'rs', ms=15, lw=linewidth)
    # arrowLoc = round(nSOL/2)
    # plt.arrow(SOL[arrowLoc, 0], SOL[arrowLoc, 1],
    #           0.01 * (SOL[arrowLoc + 1, 0] - SOL[arrowLoc, 0]), 0.01 * (SOL[arrowLoc + 1, 1] - SOL[arrowLoc, 1]),
    #           color='r', head_width=0.03, head_length=0.06)
# plt.figure(figsize=(20, 20))
plt.quiver(X1, X2, X1dot, X2dot)
plt.plot(x10, x20, 'ro', ms=10.0, markeredgecolor=None)
plt.xlim([-1.0, 1.0])
plt.ylim([-1.0, 1.0])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
if copyToDist:
        filename = str('/home/koorosh/Desktop/ME7160/Exam/Midterm_1_Takehome/report/figure/' + 'phase_portrait_around_x10_0' + str(int(100*x10)) + '_x20_' + str(int(100*x10)) + '_' + str(int(
                case_numer)) + '.eps')
else:
        filename = str('phase_portrait_around_x10_' + str(int(100*x10)) + '_x20_' + str(int(100*x10)) + '_' + str(int(
                case_numer)) + '.eps')
plt.savefig(filename, bbox_inches='tight')
plt.show()

# Calculate the eigenvalues of the linearized system
y1dot_l = f1 + \
    df1_dx1 * y1 + df1_dx2 * y2

y2dot_l = f2 + \
    df2_dx1 * y1 + df2_dx2 * y2

y1dot_l = y1dot_l.subs([(x1, x10), (x2, x20), (eps, epsUsr), (k, kUsr), (m1, m1Usr), (g, gUsr), (l, lUsr)])
y2dot_l = y2dot_l.subs([(x1, x10), (x2, x20), (eps, epsUsr), (k, kUsr), (m1, m1Usr), (g, gUsr), (l, lUsr)])

print(m1Usr * epsUsr * gUsr / (kUsr * lUsr))
A = sym.Matrix([[y1dot_l.coeff('y1'), y1dot_l.coeff('y2')], [y2dot_l.coeff('y1'), y2dot_l.coeff('y2')]])
print(sym.pretty(A))
print(sym.pretty(A.eigenvals()))
# pdb.set_trace()
#
# fid = open('equation.txt', 'w')
# fid.write(sym.latex(A.eigenvals(), mode='equation'))
# fid.close()