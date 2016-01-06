
#import numpy as np
import scipy as sp
import scipy.integrate
import matplotlib.pyplot as plt

# More plotting stuff
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation

# Needed for sliders that I use. 
import IPython.core.display as ipcd
from ipywidgets.widgets.interaction import interact, interactive

# These make vector graphics... higher quaility. If it doesn't work, comment these and try the preceeding. 

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)



def hi_res():
    try:
        # Try vector graphic first
        from IPython.display import set_matplotlib_formats
        set_matplotlib_formats( 'svg')
    except:
        # if that fails, at least raise the resolution of the plots. 
        import matplotlib as mpl
        mpl.rcParams['savefig.dpi'] = 200
        
def low_res():
    # Try vector graphic first
    from IPython.display import set_matplotlib_formats
    import matplotlib as mpl
    set_matplotlib_formats( 'png')
    mpl.rcParams['savefig.dpi'] = 72

def med_res():
    # Try vector graphic first
    from IPython.display import set_matplotlib_formats
    import matplotlib as mpl
    set_matplotlib_formats( 'png')
    mpl.rcParams['savefig.dpi'] = 144

#             #except:
#         # if that fails, at least raise the resolution of the plots. 
#         #import matplotlib as mpl
#         #mpl.rcParams['savefig'] = 120

def sdof_deriv(x1_x2, t, omega = 2.8284, Omega = 2, mu = 2, F = 10, angle = 10):
    """Compute the time-derivative of a SDOF system."""
    x1, x2 = x1_x2
    return [x2, -omega**2*x1-2*mu*x2+F*sp.cos(Omega*t)]


def solve_sdof(max_time=10.0, omega = 2.8284, Omega = 2, mu = 2, F = 10, elevation = 30, angle = 10, x0 = 1, v0 = 0, plotnow = 1):

    
    def sdof_deriv(x1_x2, t, omega = omega, Omega = Omega, mu = mu, F = F, angle = angle):
        """Compute the time-derivative of a SDOF system."""
        x1, x2 = x1_x2
        return [x2, -omega**2*x1-2*mu*x2+F*sp.cos(Omega*t)]

    x0i=((x0, v0))
    # Solve for the trajectories
    t = sp.linspace(0, max_time, int(250*max_time))
    x_t = sp.integrate.odeint(sdof_deriv, x0i, t)
    
    x, v = x_t.T

    if plotnow == 1:
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1], projection='3d')
        plt.plot(x, v, t,'--')
        plt.xlabel('x')
        plt.ylabel('v')
        ax.set_zlabel('t')

        ax.view_init(elevation, angle)
        plt.show()

    return t, x, v

def cubic_deriv(x1_x2, t, mu = .1):
    """Compute the time-derivative of a SDOF system."""
    x1, x2 = x1_x2
    return [x2, -x1-mu* x2**3]


func = cubic_deriv

def phase_plot(func, max_time=1.0, numx = 10, numv = 10, spread_amp = 1.25, args = (), span = (-1,1,-1,1)):
    """Plot the phase plane plot of the function defined by func.
    
    Parameters
    -----------
    func : string like
           name of function providing state derivatives
    max_time : float, optional
               total time of integration
    numx, numy : floats, optional
                 number of starting points for the grid of integrations
    spread_amp : float, optional
                 axis "growth" for plotting, relative to initial grid
    args : float, other
           arguments needed by the state derivative function
    span : 4-tuple of floats, optional
           (xmin, xmax, ymin, tmax)
    """
    x = sp.linspace(span[0], span[1], numx)
    v = sp.linspace(span[2], span[3], numv)
    x0, v0 = sp.meshgrid(x, v)
    x0.shape = (numx*numv,1) # Python array trick to reorganize numbers in an array
    v0.shape = (numx*numv,1)
    x0 = sp.concatenate((x0, v0), axis = 1)
    N = x0.shape[0]
    # Solve for the trajectories
    t = sp.linspace(0, max_time, int(250*max_time))
    x_t = sp.asarray([sp.integrate.odeint(func, y0 = x0i, t = t, args = args)
                      for x0i in x0])

    for i in range(N):
        x, v = x_t[i,:,:].T            
        line, = plt.plot(x, v,'-')
        #Let's plot '*' at the end of each trajectory.
        plt.plot(x[-1],v[-1],'^')
    plt.grid('on')
    xrange = (span[1]-span[0])/2
    xmid = (span[0]+span[1])/2
    yrange = (span[3]-span[2])/2
    ymid = (span[3]+span[2])/2
    plt.axis((xmid-spread_amp*xrange,xmid+spread_amp*xrange,ymid-spread_amp*yrange,ymid+spread_amp*yrange))
    #print(plt.axis())
    head_length = .1*sp.sqrt(xrange**2+yrange**2)
    head_width = head_length/3
    for i in range(N):
        x, v = x_t[i,:,:].T
        if abs(x[-1]-x[-2]) > 0 or abs(v[-1]-v[-2]) > 0:
            dx = x[-1]-x[-2]
            dv = v[-1]-v[-2]
            length = sp.sqrt(dx**2+dv**2)
            delx = dx/length*head_length
            delv = dv/length*head_length
            #plt.arrow(x[-1],v[-1],(x[-1]-x[-2])/1000,(v[-1]-v[-2])/1000, head_width=head_width, head_length = head_length, fc='k', ec='k', length_includes_head = True, width = 0.0)#,'-')
            #plt.annotate(" ", xy = (x[-1],v[-1]),xytext = (x[-1]-delx,v[-2]-delv),arrowprops=dict(facecolor='black',width = 2, frac = .5))
        plt.plot(x[0],v[0],'.')

        
    return line

def flow_plot(func, max_time=1.0, x0 = sp.array([[-1, -.9 , -0.9, -1, -1]]).T, v0 = sp.array([[1, 1, .9, 0.9, 1]]).T, args = ()):
    """Plot the flow plot of the function defined by func.

    Parameters
    -----------
    func : string like
           name of function providing state derivatives
    max_time : float, optional
               total time of integration
    x0, v0 : floats arrays, optional
                 initial values
    args : float, other
           arguments needed by the state derivative function
    """
    x0 = sp.concatenate((x0, v0), axis = 1)
    N = x0.shape[0]
    # Solve for the trajectories
    t = sp.linspace(0, max_time, int(250*max_time))
    x_t = sp.asarray([sp.integrate.odeint(func, y0 = x0i, t = t, args = args)
                      for x0i in x0])

    for i in range(N):
        x, v = x_t[i,:,:].T            
        plt.plot(x, v,'-')
        plt.plot(x[0],v[0],'ok')
        plt.plot(x[-1],v[-1],'ok')

    plt.plot(x_t[:,0,0],x_t[:,0,1],'k-')
    plt.plot(x_t[:,len(t)-1,0],x_t[:,len(t)-1,1],'k-');

    plt.grid('on')

    return

def hopf_deriv(x1_x2, t, omega = 1.0, mu =  0.1, alpha = -1.0):
    """Compute the time-derivative of a SDOF system."""
    x, y = x1_x2
    return [mu * x - omega * y + alpha * x * ( x**2 + y**2), mu * y + omega * x + alpha * y * ( x**2 + y**2)]


def duff_bif_diag(mu = .01, k_int = 0.011, alpha = .2, sigma = 0.05, amax = 10):

    full_data = False
    a = sp.linspace(0,amax,500)
    asq = a**2.0
    # equation 2.3.36
    ksq = (mu**2+(sigma-3./8.*alpha*asq)**2)*asq
    k = sp.sqrt(ksq)
    # kdif is nominally the slope of k with respect to a
    # There is no point in dividing by delta a as we only 
    # need to find the turning points. 
    kdif = k[1:-1] - k[0:-2]

    soln_1 = 0
    soln_2 = 0
    soln_3 = 0
    # Let's try looking for an inflection point (change of sign).
    # If there is none, then just plot the amplotude of the response versus the
    # amplotude of the excitation. 
    try:
        # This is a bit of Python trickery that I Googled. It's used repeatedly.
        # It finds the first location where the slope of k(a) turns negative.
        first_inflection = next(i for i, kdif in enumerate(kdif) if kdif < 0.0)
        kfi = k[first_inflection]

        # Try to find a second inflection point. If there is one, there should be a second
        # however, it may not be findable within the range of a provided.
        # If it's not found, put out some information and a less detailed plot that shows the single inflection point. 
        try:
            second_inflection = first_inflection + next(i for i, kdif in enumerate(kdif[first_inflection:]) if kdif > 0.0)
            full_data = True
        except:
            print('Plot range is incomplete. Try increasing amax')
            plt.plot(k[first_inflection],a[first_inflection],'o')
            plt.annotate('B', xy = (k[first_inflection]+.0005,a[first_inflection]))
            
        ksi = k[second_inflection]
        
        # Same trick, but looking for points on curve that pair with the inflection
        # points. 
        second_land = next(i for i, k in enumerate(k) if k > ksi)
        first_land = second_inflection + next(i for i, k in enumerate(k[second_inflection:]) if k > kfi)

        # Plot the important points and label them. 
        plt.plot(k[first_inflection],a[first_inflection],'o')
        plt.annotate('B', xy = (k[first_inflection]+.0005,a[first_inflection]))
        plt.plot(k[second_inflection],a[second_inflection],'o')
        plt.annotate('E', xy = (k[second_inflection]-.001,a[second_inflection]))
        plt.plot(k[second_land],a[second_land],'o')
        plt.annotate('H', xy = (k[second_land]+.0005,a[second_land]-.02))
        plt.plot(k[first_land],a[first_land],'o')
        plt.annotate('C', xy = (k[first_land],a[first_land]+.02))

        # Find the solutions of interest (k_int)
        soln_1 =  next(i for i, k in enumerate(k) if k > k_int)

        # Plot the lines
        plt.plot(k[:first_inflection],a[:first_inflection],'-')
        plt.plot(k[first_inflection:second_inflection],a[first_inflection:second_inflection],'--')
        plt.plot(k[second_inflection:],a[second_inflection:],'-')

        if soln_1 > first_inflection or k[soln_1] < k[second_inflection]:
            plt.plot(k[soln_1],a[soln_1],'*')
        else:
            soln_2 =  first_inflection + next(i for i, k in enumerate(k[first_inflection:]) if k < k_int)
            soln_3 =  second_inflection + next(i for i, k in enumerate(k[second_inflection:]) if k > k_int)
            plt.plot(k[soln_1],a[soln_1],'*b')
            plt.plot(k[soln_2-1],a[soln_2-1],'*g')
            plt.plot(k[soln_3],a[soln_3],'*r')
            #plt.annotate('One of the possible amplitudes', xy = (k[soln_3],a[soln_3]))

    except:
        plt.plot(k,a,'-')

        
    # Labels and grid
    plt.title('Response amplitude versus driving force amplitude.')
    plt.xlabel('k')
    plt.ylabel('a')
    plt.grid('on')
    
    # Find (and return) locations of fixed points

    gamma_1 = sp.arctan2( mu*a[soln_1]/k[soln_1],(-sigma*a[soln_1]+3./8.*alpha*a[soln_1]**3)/k[soln_1])
    if gamma_1 < 0:
        gamma_1 = gamma_1 + sp.pi

    if soln_2 == 0:
        soln_2 = 1
        gamma_2 = 0
    else:
        gamma_2 = sp.arctan2( mu*a[soln_2]/k[soln_2],(-sigma*a[soln_2]+3./8.*alpha*a[soln_2]**3)/k[soln_2])
        if gamma_2 < 0.0:
            gamma_2 = gamma_2 + sp.pi
          
    if soln_3 == 0:
        soln_3 = 1
        gamma_3 = 0
    else:
        gamma_3 = sp.arctan2( mu*a[soln_3]/k[soln_3],(-sigma*a[soln_3]+3./8.*alpha*a[soln_3]**3)/k[soln_3])
        if gamma_3 < 0.0:
            gamma_3 = gamma_3 + sp.pi

    return [a[soln_1],a[soln_2-1],a[soln_3]], [gamma_1, gamma_2, gamma_3]
    

def duf_bif_deriv(x1_x2, t, mu = 0.01, k = 0.013, alpha = .2, sigma = 0.05):
    x1, x2 = x1_x2
    return [-mu * x1 + k * sp.sin(x2), sigma - 3./8.*alpha*x1**2+k*sp.cos(x2)/(x1)]

def duff_bif_phase(k = 0.012):
    plt.subplot(1,2,1)
    mu = 0.01; alpha = .2; sigma = 0.05; amax = 1.0
    phase_plot(duf_bif_deriv, max_time=100.0, numx = 7, numv = 7, span = (0.01,1.2,-0.5,4), args = (mu, k, alpha, sigma))
    plt.xlabel('a')
    plt.ylabel('$\gamma$')
    plt.subplot(1,2,2)
    a, gamma = duff_bif_diag(mu = 0.01, k_int = k, alpha = alpha, sigma = sigma, amax = amax)
    plt.title('')
    plt.subplot(1,2,1)
    sp.reshape(a,(3,1))
    plt.plot(sp.reshape(a,(1,3)),sp.reshape(gamma,(1,3)),'*')
    plt.tight_layout(w_pad=1.5)

# This little bit makes the prior function interactive with 'ipcd.display(duff_interact_jump)'
duff_interact_jump = interactive(duff_bif_phase,  k = (0.0070,0.017,.0004))


def duff_amp_solve(mu = 0.01, k = 0.013, alpha = .2, sigma = (-0.5,.5)):
    sigma = sp.linspace(sigma[0],sigma[1],1000)
    #print(sigma.size)
    #print(sigma)
    a = sp.zeros((sigma.size,3))*1j
    first = 1
    #print(a)
    for idx, sig in enumerate(sigma):
        #print(idx)
        
        p = sp.array([alpha**2, 0, -16./3.*alpha*sig, 0, 64./9.*(mu**2 + sig**2),0,-64./9.*k**2])
        soln = sp.roots(p)
        #print('original soln')
        #print(soln)
        #print(soln)
        #print(sp.sort(soln)[0:5:2])
        sorted_indices = sp.argsort(sp.absolute(soln))
        a[idx,:] = soln[sorted_indices][0:5:2]
        if sum(sp.isreal(a[idx,:])) == 3 and first == 1:
            first = 0
            #if sp.absolute(a[idx,2] - a[idx,1]) < sp.absolute(a[idx,1] - a[idx,0]):
            solns = sp.sort(sp.absolute(a[idx,:]))
            #print(solns)
            if (solns[2] - solns[1]) > (solns[1]-solns[0]):
                ttl = 'Hardening spring'
                softening = False
            else:
                ttl = 'Softening spring'
                softening = True
                #print(solns)
                
            first_bif_index = idx

        if first == 0 and sum(sp.isreal(a[idx,:])) == 1:
            first = 2
            second_bif_index = idx
                
    if softening == True:

        low_sig = sigma[0:second_bif_index]

        low_amp = sp.zeros(second_bif_index)
        low_amp[0:first_bif_index] = sp.absolute(sp.sum(sp.isreal(a[0:first_bif_index,:])*a[0:first_bif_index,:],axis = 1))
        low_amp[first_bif_index:second_bif_index] = sp.absolute(a[first_bif_index:second_bif_index,:]).min(axis = 1)

        med_sig = sigma[first_bif_index:second_bif_index]

        med_amp = sp.sort(sp.absolute(a[first_bif_index:second_bif_index,:]),axis = 1)[:,1]
        
        high_sig = sigma[first_bif_index:]
        high_amp = sp.zeros(sigma.size - first_bif_index)
        high_amp[0:second_bif_index - first_bif_index] = sp.absolute(a[first_bif_index:second_bif_index,:]).max(axis = 1)
        high_amp[second_bif_index - first_bif_index:] = sp.absolute(sp.sum(sp.isreal(a[second_bif_index:,:])*a[second_bif_index:,:],axis = 1))
        
    else:
        
        high_sig = sigma[0:second_bif_index]

        high_amp = sp.zeros(second_bif_index)
        high_amp[0:first_bif_index] = sp.absolute(sp.sum(sp.isreal(a[0:first_bif_index,:])*a[0:first_bif_index,:],axis = 1))
        high_amp[first_bif_index:second_bif_index] = sp.absolute(a[first_bif_index:second_bif_index,:]).max(axis = 1)

        med_sig = sigma[first_bif_index:second_bif_index]

        med_amp = sp.sort(sp.absolute(a[first_bif_index:second_bif_index,:]),axis = 1)[:,1]
        
        low_sig = sigma[first_bif_index:]
        low_amp = sp.zeros(sigma.size - first_bif_index)
        low_amp[0:second_bif_index - first_bif_index] = sp.absolute(a[first_bif_index:second_bif_index,:]).min(axis = 1)
        low_amp[second_bif_index - first_bif_index:] = sp.absolute(sp.sum(sp.isreal(a[second_bif_index:,:])*a[second_bif_index:,:],axis = 1))
        
    plt.plot(low_sig,low_amp,'-b')
    plt.plot(med_sig, med_amp, '--g')
    plt.plot(high_sig,high_amp,'-r')

    plt.title(ttl)
    plt.xlabel('$\sigma$')
    plt.ylabel('a')
    return 

def solve_henon(alpha = 0.2, beta = 0.3, firstindex = 0, numsteps = 10, x0 = 1.0, y0 = 0.0):
    if firstindex > 0:
        firstindex = 0
    if -firstindex > numsteps:
        numsteps = -firstindex
    numsteps = numsteps + 1
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    x = sp.zeros(numsteps)
    y = sp.zeros(numsteps)
        
    x[-firstindex] = x0
    y[-firstindex] = y0

    def henon_F(x, y, alpha = alpha, beta = beta):
        return [1.0 + y - alpha * x**2, beta * x]
    
    def henon_F_inv(x, y, alpha = alpha, beta = beta):
        return [y / beta, x - 1 + (alpha / (beta**2)) * y**2]
    
    for i in range(-firstindex + 1, numsteps):
        x[i], y[i] = henon_F(x[i-1], y[i-1], alpha, beta)

    for i in range(-firstindex - 1, 0 - 1, -1):
        x[i], y[i] = henon_F_inv(x[i+1], y[i+1], alpha, beta)
  
    plt.plot(x,y,'*')
    
    for i in range(numsteps):
        #ax.annotate('%s' % i, xy=[x[i],y[i]], textcoords='offset points')
        ax.annotate('{}'.format(i + firstindex), xy=(x[i],y[i]), xytext=(3, -14), ha='right',textcoords='offset points')
    
    xrange = max(x) - min(x)
    yrange = max(y) - min(y)
    buf = 0.1
    ax.set_xlim((min(x) - buf * xrange, max(x) + buf * xrange))
    ax.set_ylim((min(y) - buf * yrange, max(y) + buf * yrange))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid('on')


    return

def par_duff_bif(sigma = 2.0, alpha = 2.0, mu = 0.5):
    #All parameters set to one for this simple illustration.
    k2 = sp.sqrt(sigma**2 + 4*mu**2)
    
    k = sp.linspace(0.0, 2.0 * k2, 1000)
    #print(k)
    ahigh = sp.sqrt(4/3/alpha*(sigma+sp.sqrt(k**2-4*mu**2)))
    #print(ahigh)
    mask_high = sp.isreal(ahigh)
    plt.plot(k[mask_high],sp.real(ahigh[mask_high]),'-k')

    alow = sp.sqrt(4/3/alpha*(sigma-sp.sqrt(k**2-4*mu**2)))
    mask_low = sp.isreal(alow)
    plt.plot(k[mask_low],sp.real(alow[mask_low]),'--k')

    plt.plot(k[k<k2],k[k<k2]*0,'-k')
    plt.plot(k[k>k2],k[k>k2]*0,'--k')

    plt.xlabel('$k$')
    plt.ylabel('a')
    plt.grid('on')
    plt.annotate('$k_2$', xy=(k2+.03,0.03))
    plt.title('Bifurcation Diagram using $k$ as a control parameter.');
    plt.axis([0,2.0*k2,-0.1,sp.ceil(max(sp.real(ahigh)))]);

def par_duff_deriv(x_v, t, eps = 0.1, mu = 1.0, alpha = 1.0, k = 3.5, sigma = 1.0):
    x, v = x_v
    return [v, -x - eps*(2*mu*v+alpha*x**3+2*k*x*sp.cos((2+eps*sigma)*t))]
def par_duff_phase(k=2):
    phase_plot(par_duff_deriv, max_time = 100, numx = 1, numv = 2, args=(0.1, 1.0, 1.0, k, 1.0))
    plt.axis('auto')
    return


def quad_damp_deriv(x1_x2, t, omega = 4, epsilon = .1):
        """Compute the time-derivative of a SDOF system."""
        x1, x2 = x1_x2
        return [x2, -omega**2*x1-epsilon*x2*sp.absolute(x2)]

def quad_decay_plot(x0 = 1, v0 = 1, max_time = 54, omega = 1, epsilon = 0.1):
    x0i=((x0, v0))
    # Solve for the trajectories
    t = sp.linspace(0, max_time, int(250*max_time))


    x_t = sp.integrate.odeint(quad_damp_deriv, x0i, t, args = (omega, epsilon))
    #x, t = x_t
    a0 = sp.sqrt(x0**2+v0**2/omega**2)
    plt.plot(t,x_t[:,0],'-',t,a0/(1+(4*epsilon*omega*a0/3/sp.pi)*t),'--g',t,-a0/(1+(4*epsilon*omega*a0/3/sp.pi)*t),'--g')
    plt.grid('on')

def lin_poincare(tfinal = 1):
    t = sp.arange(0,tfinal,0.001)
    #x = sp.exp(-0.02*t)*(-0.5*sp.cos(2*t)-1.5 *sp.sin(2*t))+0.5*sp.cos(2*t)+sp.sin(2*t)
    A = 5;B = 5; omega_n = 10; omega = 20; zeta = .02;omega_d = omega_n*sp.sqrt(1-zeta**2)

    x = A * sp.sin(omega*t)+B*sp.exp(-zeta*omega_n*t)*sp.cos(omega_d*t)
    y = A*omega*sp.cos(omega*t)-B*omega_d*sp.exp(-zeta*omega_n*t)*sp.sin(omega_d*t)-B*omega_n*zeta*sp.exp(-omega_n*zeta*t)*sp.cos(omega_d*t)
    #y = -0.02*sp.exp(-0.02*t)*(sp.cos(2*t)-2 *sp.sin(2*t))+2*sp.cos(2*t)-sp.sin(2*t)
    t2 = sp.arange(0,tfinal,2*sp.pi/omega*1.001)
    x2 = A * sp.sin(omega*t2)+B*sp.exp(-zeta*omega_n*t2)*sp.cos(omega_d*t2)
    y2 = A*omega*sp.cos(omega*t2)-B*omega_d*sp.exp(-zeta*omega_n*t2)*sp.sin(omega_d*t2)-B*omega_n*zeta*sp.exp(-omega_n*zeta*t2)*sp.cos(omega_d*t2)
    plt.plot(x,y,'-',x2,y2,'.r')
    plt.title('Poincar\\\'e plot on phase plane, slight error in period estimation.')
    plt.xlabel('x')
    plt.ylabel('y')
