from __future__ import print_function, division

import numpy as np
from transit import Central, System, Body
import batman
import math

R_sun = 69550800000.0 #astropy.constants.R_sun.cgs.value
M_sun = 1.9891e+33 #astropy.constants.M_sun.cgs.value
big_G = 6.67384e-08 #astropy.constants.G.cgs.value
daytosecond = 86400.

def dilution_samples(s, which='A', band='Kepler'):
    """
    Returns dilution samples in band, according to StarModel s

    :param s:
        :class:`BinaryStarModel` Object, (defined in ``isochrones`` module)

    :param which:
        'A' for primary, 'B' for secondary.

    :param band:
        Photometric bandpass of interest.
    """

    mA = s.samples['{}_mag_A'.format(band,which)]
    mB = s.samples['{}_mag_B'.format(band,which)]
    
    fA = 10**(-0.4*mA)
    fB = 10**(-0.4*mB)
    if which=='A':
        return fB/(fB+fA)
    elif which=='B':
        return fA/(fB+fA)
    else:
        raise ValueError('Invalid choice: {}'.format(which))

def density_samples(s, which='A'):
    """
    Returns density samples according to StarModel

    :param s:
        :class:`StarModel` object.

    :param which:
        'A' for primary, 'B' for secondary.
    """
    if which=='A':
        if 'mass_A' in s.samples:
            m = s.samples['mass_A'.format(which)]
        else:
            m = s.samples['mass'.format(which)]
        r = s.samples['radius'.format(which)]
    elif which=='B':
        m = s.samples['mass_B']
        r = s.samples['radius_B']
    else:
        raise ValueError('Invalid choice: {}'.format(which))
    return 0.75*m*M_sun / (np.pi * (r*R_sun)**3)

def t_folded(t, per, ep):
    return (t + per/2 - ep) % per - (per/2)

def transit_lc(p, t, texp=None): 
    """
    Returns flux using :transit: at given times, given parameters:

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
        
    n_planets = (len(p) - 4)//6#number of planets based on input param array
    
    rhostar, q1, q2, dilution = p[:4] #assigning star's params from input

    # u1 = 2*math.sqrt(q1)*q2 #convert q1,q2 to u1,u2 from Kipping (2013)
    # u2 = math.sqrt(q1)*(1.-2*q2)

    central = Central(q1=q1, q2=q2) #setting the central body of the system
    central.density = rhostar
    s = System(central, dilution=dilution)


    for i in range(n_planets): #iteratively adds the planets passed in from params
        period, epoch, b, rprs, e, w = p[4+i*6:10+i*6]
        r = central.radius * rprs
        body = Body(radius=r, mass=0, period=period, t0=epoch,
                   e=e, omega=w, b=b)
        s.add_body(body)


    return s.light_curve(t, texp=texp) #returns a numpy array of flux


def batman_lc(p,t,initial_model,max_err=None,texp=None):
    """
    Returns flux using :batman: at given times, given parameters:

    :param p:
        Parameter vector, of length 4 + 6*Nplanets
        p[0:4] = [rhostar, q1, q2, dilution]
        p[4+i*6:10+i*6] = [period, epoch, b, rprs, e, w] for i-th planet

    :param t:
        Times at which to evaluate model.

    :param max_err:
        (Unimplemented) Maximum error in flux used to calculate supersample factor.

    :param texp:
        Exposure time.  If not provided, assumed to be median t[1:]-t[:-1]

    """

    if texp is None: #if we aren't given an exposure time, calculate it
        texp = np.median(t[1:] - t[:-1])

    model = initial_model

    n_planets = (len(p) - 4)//6#number of planets based on input param array
    
    rhostar, q1, q2, dilution = p[:4] #assigning star's params from input

    u1 = 2*math.sqrt(q1)*q2 #convert q1,q2 to u1,u2 from Kipping (2013)
    u2 = math.sqrt(q1)*(1.-2*q2)

    totaldelta = np.zeros_like(t) #difference from zero at all points

    for i in range(n_planets): #assigning parameters, once for each planet

        #calculate the corresponding a/R_star from rhostar and period
        periodcgs = p[i*6+4]*daytosecond #convert the period to cgs
        ars = ((rhostar*big_G*periodcgs**2)/(3*math.pi))**(1./3.) #a over R_star, from Winn (2014) eqn (30)

        #calculate inc from b, a/R_star, e, w
        b = p[6+i*6]
        e = p[8+i*6]
        w = p[9+i*6]

        #from Winn (2014), eqn 7
        incl = math.acos((b/ars)*((1+e*math.sin(w))/(1-e**2))) *180./math.pi
        w = w * 180./math.pi

        #assigning parameters
        params = batman.TransitParams()
        params.per = p[4+i*6]
        params.t0 = p[5+i*6]
        rp = p[7+i*6]
        params.rp = rp
        params.a = ars
        params.inc = incl
        params.ecc = e
        params.w = w
        params.limb_dark = 'quadratic'
        params.u = [u1,u2]

        # #delta = rp**2
        # T_0 = p[i*6+4]/(ars*math.pi)
        # ingressduration = (T_0*rp)/math.sqrt(1-b**2)
        # #sigma = max_err
        # #I = exp_time
        # N = ((texp*rp**2)/(8.*ingressduration*max_err))**(0.5)
        # print(max_err,T_0,ingressduration,N)



        #times = t
        #building the model and light curve
        flux = model.light_curve(params)

        #implementing the dilution and adding to the total light curve data
        totaldelta += (1-flux)*(1-dilution)

    #returning the cumulative light curve for all planets
    return (1-totaldelta)







        