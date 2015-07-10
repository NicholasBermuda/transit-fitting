from __future__ import print_function, division

import numpy as np
from transit import Central, System, Body #we're using these classes from transit

def t_folded(t, per, ep):
    return (t + per/2 - ep) % per - (per/2)

def lc_eval(p, t, texp=None): #returns an numpy ndarray light curve
                              #using the simple layout of transit package
    """
    Returns flux at given times, given parameters.

    :param p:
        Parameter vector, of length 4 + 6*Nplanets
        p[0:4] = [rhostar, q1, q2, dilution]
        p[4+i*6:10+i*6] = [period, epoch, b, rprs, e, w] for i-th planet

    :param t:
        Times at which to evaluate model.

    :param texp:
        Exposure time.  If not provided, assumed to be median t[1:]-t[:-1]

    """
    if texp is None: #if we aren't given an exposure time, calculate it
        texp = np.median(t[1:] - t[:-1])
        
    n_planets = (len(p) - 4)//6 #number of planets based on input param array
    
    rhostar, q1, q2, dilution = p[:4] #assigning star's params from input

    central = Central(q1=q1, q2=q2) #setting the central body of the system
    central.density = rhostar
    s = System(central, dilution=dilution)

    tot = 0 #vestigial?
    close_to_transit = np.zeros_like(t).astype(bool) #vestigial?

    for i in range(n_planets): #iteratively adds the planets passed in from params
        period, epoch, b, rprs, e, w = p[4+i*6:10+i*6]
        r = central.radius * rprs
        body = Body(flux=0, r=r, mass=0, period=period, t0=epoch,
                   e=e, omega=w, b=b)
        s.add_body(body)

        tfold = t_folded(t, period, epoch) #where does this go?

    return s.light_curve(t, texp=texp) #returns a numpy array of flux
        
