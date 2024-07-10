# Kieran Owens 2024
# Code to generate flows and maps with time-varying parameters

# Contents:
# 1. Dependencies
# 2. Process class
# 3.1 Integration function
# 3.2 Map computing function
# 4. Parameter variation function
# 5. Define flows
# 6. Define maps

# Available flows:
# * ProcessLorenz
# * ProcessAizawa

# Available maps:
# * ProcessLogistic
# * ProcessSin

##############################################################################
# 1. Dependencies
##############################################################################

import numpy as np
from scipy.integrate import odeint, solve_ivp
from dataclasses import dataclass

##############################################################################
# 2. Process class
##############################################################################

@dataclass
class Process:
    name: str       # process name, e.g. lorenz
    type: str       # process type, e.g. flow
    params: tuple   # parameters for process
    init: list      # initial conditions
    eq: object      # function definition for process
    eq_ivp: object  # function definition for use with solve_ivp


##############################################################################
# 3.1 Integration function
##############################################################################

def integrateODE(process, t, p=None, y0=None, method=None, verbose=False):
    """
    Integrate a system of ODEs

    Parameters
    ----------
    process : Process
        A dynamical process provided as a custom 'Process' dataclass
        that includes data fields for name (str), type (str), parameters (tuple),
        initial conditions (list), and equations (a function object)

    t : list or iterable (e.g. np.arange(...)))
        The time points at which the process will be sampled during integration.

    p : tuple of function objects
        An alternative tuple of time-varying parameters, e.g. each passed as
        a function with the format lambda t: f(t) (where f(t) can be constant).
        Default: None, in which case process.params will be used.

    y0: list
        A list of initial condition coordinates, e.g. [1.0, 1.0, 1.0]
        Default: None, in which case process.init will be used.

    method: str
        Specify the integration method. Default: None which uses
        scipy.integrate.odeint LSODE. Otherwise specify 'RK45' to use
        Runge-Katta via scipy.integrate.solve_ivp.

    verbose: boolean
        Determines whether a message is provided to the user

    Returns
    -------
    The array output of an ODE integrator.
    """

    # set initial conditions
    if y0 == None:
        y0 = process.init

    # parameter
    if p == None:
        p = process.params

    # user message if verbose is selected
    if verbose:
        print(f"Integrating process '{process.name} {process.type}' with init {y0} and params {p}")

    # integrate and return the result
    if method == None:

        return odeint(process.eq, y0, t, p)
    
    else:

        return solve_ivp(process.eq_ivp, t_span=(np.min(t), np.max(t)), y0=y0,
                         method=method, t_eval=t, args=p).y.T
    
##############################################################################
# 3.2 Map computing function
##############################################################################
    
def computeMap(process, t, p=None, y0=None, verbose=False):
    """
    Compute a map process.

    Parameters
    ----------
    process : Process
        A dynamical process provided as a custom 'Process' dataclass
        that includes data fields for name (str), type (str), parameters (tuple),
        initial conditions (list), and equations (a function object)

    t : list or iterable (e.g. np.arange(...)))
        The time points over which the process will be computed.

    p : tuple of function objects
        An alternative tuple of time-varying parameters, e.g. each passed as
        a function with the format lambda t: f(t) (where f(t) can be constant).
        Default: None, in which case process.params will be used.

    y0: list
        A list of initial condition coordinates, e.g. [1.0, 1.0, 1.0]
        Default: None, in which case process.init will be used.

    verbose: boolean
        Determines whether a message is provided to the user

    Returns
    -------
    The array output of the computed map.
    """

     # set initial conditions
    if y0 == None:
        y0 = process.init

    # parameter
    if p == None:
        p = process.params

    # user message if verbose is selected
    if verbose:
        print(f"Computing '{process.name} {process.type}' with init {y0} and params {p}")

    solution = np.zeros((len(t), len(y0)))
    solution[0,:] = y0
    for i in t[1:]:
        solution[i,:] = process.eq(solution[i-1,:], i, *p)

    return solution

##############################################################################
# 4. Parameter variation function
##############################################################################

def get_parameters(process, time, param_idx, type, prop):
    """
    Given a set of default parameters for a dynamical process, update one of
    the parameters to have a specific time-varying functional form.

    Parameters
    ----------
    process : Process
        A dynamical process provided as a custom 'Process' dataclass
        that includes data fields for name (str), type (str), parameters (tuple),
        initial conditions (list), and equations (a function object)

    time : list or iterable (e.g. np.arange(...)))
        The time points at which the process will be sampled during integration.

    param_idx : int
        The index of the parameter to be altered within the Process object.

    type: str
        Specification of the functional form of the new time-varying parameter.
        Options include 'linear', and 'sin {periods}'.

    prop: float
        The proportion by which to vary the selected parameter. E.g., for a
        parameter with a default value of 10, prop 0.2 will vary the parameter
        between 0.8 and 1.2 based on the specified funcitonal form.

    Returns
    -------
    A tuple of function objects (e.g. lambda t: f(t)) each of which defines
    the behaviour of a time-varying parameter.
    """

    # initialise params
    p = [param for param in process.params]

    # max time value
    tmax = np.max(time)
    tsteps = time.size

    # default valu eof the parameter to vary
    value = p[param_idx](0)
    prop = value * prop

    # create new parameter depending on the specified type
    # linear function from value - prop to value + prop from t=0 to t=tmax
    if type == 'linear':

        p[param_idx] = lambda t: (value - prop) + (t/tmax) * 2 * prop

    # sinusoidal variation for a specified number of periods
    # input as 'sin {periods}'
    elif 'sin' in type:

        periods = float(type.split()[1])
        
        p[param_idx] = lambda t: value + prop * np.sin(2 * np.pi * periods * (t/tmax))

    return tuple(p), p[param_idx](time)

##############################################################################
# 5. Define flows
##############################################################################

################
# Lorenz
################

def lorenz(state, t, sigma, beta, rho):

    x, y, z = state
     
    dx = sigma(t) * (y - x)
    dy = x * (rho(t) - z) - y
    dz = x * y - beta(t) * z
     
    return [dx, dy, dz]

def lorenz2(t, state, sigma, beta, rho):

    x, y, z = state
     
    dx = sigma(t) * (y - x)
    dy = x * (rho(t) - z) - y
    dz = x * y - beta(t) * z
     
    return [dx, dy, dz]

ProcessLorenz = Process(name = 'lorenz',
    type = 'flow',
    params = (lambda t: 10.0, lambda t: 8.0 / 3.0, lambda t: 28.0),
    init = [-9.79, -15.04, 20.53],
    eq = lorenz,
    eq_ivp = lorenz2)



################
# Aizawa
################

def langford(state, t, a, b, c, d, e, f):

    x, y, z = state
     
    dx = x*z - b(t)*x - d(t)*y
    dy = d(t)*x + y*z - b(t)*y
    dz = c(t) + a(t)*z - (1/3)*z**3 - x**2 - y**2 - e(t)*z*x**2 - e(t)*z*y**2 + f(t)*z*x**3
     
    return [dx, dy, dz]

def langford2(t, state, a, b, c, d, e, f):

    x, y, z = state
     
    dx = x*z - b(t)*x - d(t)*y
    dy = d(t)*x + y*z - b(t)*y
    dz = c(t) + a(t)*z - (1/3)*z**3 - x**2 - y**2 - e(t)*z*x**2 - e(t)*z*y**2 + f(t)*z*x**3
     
    return [dx, dy, dz]

ProcessLangford = Process(name = 'Langford',
    type = 'flow',
    params = [lambda t: 0.95, lambda t: 0.7, lambda t: 0.6, lambda t: 3.5, lambda t: 0.25, lambda t: 0.1],
    init = [-0.78450179, -0.62887672, -0.17620268],
    eq = langford,
    eq_ivp = langford2)




##############################################################################
# 6. Define maps
##############################################################################

################
# Logistic
################

def logistic(x, t, r):

    return r(t) * x * (1 - x)

ProcessLogistic = Process(name = 'logistic',
    type = 'map',
    params = tuple([lambda t: 3.6]),
    init = [0.6],
    eq = logistic,
    eq_ivp = None)


################
# Sine map
################

def sinmap(x, t, r):

    return r(t) * np.sin(x * np.pi)

ProcessSin= Process(name = 'sin map',
    type = 'map',
    params = tuple([lambda t: 3.0]),
    init = [0.6],
    eq = sinmap,
    eq_ivp = None)