from __future__ import print_function, division

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os, os.path
import math
import shutil

from scipy.special import beta

from astropy import constants as const
G = const.G.cgs.value
M_sun = const.M_sun.cgs.value
R_sun = const.R_sun.cgs.value
DAY = 86400 #in seconds


import emcee
import pymultinest

try:
    import triangle
except ImportError:
    triangle=None

from transit.transit import InvalidParameterError
    
from .utils import lc_eval

class TransitModel(object):
    """Model of one or more transiting planets around a particular star
    
    :param lc:
        LightCurve object

    :param width:
        The number of durations around a transit calculated to evaluate
        model.  Model is evaluated between ``tc - width*duration`` and
        ``tc + width*duration``, where ``tc`` is the transit center time.

    """
    def __init__(self, lc, width=2, continuum_method='constant',use_leastsq=False,use_emcee=False):

        self.lc = lc #KeplerLightCurve object
        self.width = width
        self.continuum_method = continuum_method



        self._bestfit = None
        self._samples = None

        #the following is useful for the fit wrapper
        self.use_leastsq = use_leastsq #if you want to use least squares for fitting
        self.use_emcee = use_emcee #if you want to use emcee for fitting


    def continuum(self, p, t):
        """ Out-of-transit 'continuum' model.

        :param p:
            List of parameters.  For now all that is implemented
            is a constant.

        param t:
            Times at which to evaluate model.
        
        """
        #creates an array of ones in the same shape as params
        p = np.atleast_1d(p)
        
        #and returns the flux zero-point at all times in the lc
        return p[0]*np.ones_like(t)
        
    def evaluate(self, par):
        """Evaluates light curve model at light curve times

        :param p:
            Parameter vector, of length 5 + 6*Nplanets
            p[0] = flux zero-point
            p[1:5] = [rhostar, q1, q2, dilution]
            p[5+i*6:11+i*6] = [period, epoch, b, rprs, e, w] for i-th planet

        :param t:
            Times at which to evaluate model.
            

        """
        p = [par[i] for i in range(5+6*self.lc.n_planets)] #don't slice the cube!


        #starts by creating a continuum model at times in the light curve
        f = self.continuum(p[0], self.lc.t)

        # Identify points near any transit
        close = np.zeros_like(self.lc.t).astype(bool) #intialising a boolean mask
                                                      #over all the times
        for i in range(self.lc.n_planets): #for all of the planets in the input
                                           #n_planets is from the KeplerLightCurve object
            #creates the close array by changing value of indices that are close to the
            #lowest value of the transit (width is passed into TransitModel)
            close += self.lc.close(i, width=self.width)

        #evaluates the light curve data at points where we're in transit
        f[close] = lc_eval(p[1:], self.lc.t[close], texp=self.lc.texp)
        return f

    # THIS WILL BECOME A WRAPPER FOR THE THREE TYPES OF FIT
    # def fit(self, **kwargs):
    #     """
    #     Wrapper for either :func:`fit_leastsq' or :func:`fit_emcee` or 
    #     :func:`fit_multinest`

    #     Default is to use MultiNest; set `use_emcee` or `fit_leastsq` 
    #     to `True` or call them directly if you want to use MCMC or least squares
    #     """
    #     if self.use_leastsq:

    #         #remove kwargs for emcee and multinest

    #     elif self.use_emcee:
    #         #remove kwargs for leastsq and multinest

    #     else:
    #         self.fit_multinest(**kwargs)


    def fit_leastsq(self, p0, method='Powell', **kwargs):
        #using the scipy.optimize.minimize function
        #cost is negative post because we're using minimize
        fit = minimize(self.cost, p0, method=method, **kwargs)
        self._bestfit = fit.x
        return fit

    def fit_emcee(self, p0=None, nwalkers=200, threads=1,
                  nburn=10, niter=100, **kwargs):
        #fits the parameters using the emcee package
        if p0 is None:
            p0 = self.lc.default_params

        ndim = len(p0)

        # TODO: improve walker initialization!

        nw = nwalkers
        p0 = np.ones((nw,ndim)) * np.array(p0)[None,:]

        p0[:, 0] += np.random.normal(size=nw)*0.0001 #flux zp
        p0[:, 1] += np.random.normal(size=nw) #rhostar
        p0[:, 2] += np.random.normal(size=nw)*0.1 #q1
        p0[:, 3] += np.random.normal(size=nw)*0.1 #q2
        p0[:, 4] += np.random.normal(size=nw)*0.1 #dilution

        for i in range(self.lc.n_planets):
            p0[:, 5 + 6*i] += np.random.normal(size=nw)*1e-4 #period
            p0[:, 6 + 6*i] += np.random.normal(size=nw)*0.001 #epoch
            p0[:, 7 + 6*i] = np.random.random(size=nw)*0.8 # impact param
            p0[:, 8 + 6*i] *= (1 + np.random.normal(size=nw)*0.01) #rprs
            p0[:, 9 + 6*i] = np.random.random(size=nw)*0.1 # eccentricity
            p0[:, 10 + 6*i] = np.random.random(size=nw)*2*np.pi

        p0 = np.absolute(p0) # no negatives allowed 
                                                
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self, threads=threads)

        pos,prob,state = sampler.run_mcmc(p0, nburn)
        sampler.reset()

        sampler.run_mcmc(pos, niter)

        self.sampler = sampler
        return sampler

    def fit_multinest(self,n_live_points = 1000,basename='chains/1-', verbose=True,overwrite=True,**kwargs):

        self._mnest_basename = str(self.lc.koinum) + '/' + basename

        #if overwrite and os.path.exists(str(self.lc.koinum)): shutil.rmtree(str(self.lc.koinum))

        #creates the directory for the output
        folder = os.path.abspath(os.path.dirname(self._mnest_basename))
        if not os.path.exists(self._mnest_basename):
            os.makedirs(self._mnest_basename)


        if hasattr(self,'which'): self.n_params = 9 + 6*self.lc.n_planets
        else: self.n_params = 5 + 6*self.lc.n_planets

        pymultinest.run(self.mnest_loglike,self.mnest_prior,self.n_params,
                            n_live_points=n_live_points,outputfiles_basename=self._mnest_basename,verbose=verbose,**kwargs)

        self._make_samples()

    def mnest_prior(self, cube, ndims, nparams):
        """
        Transforms the unit cube into parameter cube

        Uses simple flat priors, more complicated transormations will occur in lnprior

        Priors are [flux_zp, rhostar, q1, q2, dilution] 
        and [period, epoch, b, rprs, ecc, omega] for each planet
        """
        #flat priors for fluxzp,rhostar,q1,q2,dilution
        cube[0] = 0.04*cube[0] + 0.98 #flux_zp in [0.98,1.02)
        cube[1] = 199.999*cube[1] + 1e-4 #rhostar in [1e-4, 200)
        # cube[2] = unchanged # q1 in [0,1)
        # cube[3] = unchanged # q2 in [0,1)
        # cube[4] = unchanged # dilution in [0,1)

        counter = 5
        for i in xrange(self.lc.n_planets): #iterating over each planet in the light curve system
            #grabbing these prior values as the mean and error for flat priors
            prior_p_mu, prior_p_sig = self.lc.planets[i]._period
            prior_ep_mu, prior_ep_sig = self.lc.planets[i]._epoch
            #setting the flat priors between mu +- 10sigma for period, epoch
            cube[counter] = 20*prior_p_sig*cube[counter] + prior_p_mu - 10*prior_p_sig #period
            cube[counter+1] = 20*prior_ep_sig*cube[counter+1] + prior_ep_mu - 10*prior_ep_sig #epoch
            cube[counter+2] = 2*cube[counter+2] #b in [0,2)
            cube[counter+3] = 0.295*cube[counter+3] + 0.005 #rprs in [0.005,0.3)
            # cube[counter+4] = unchanged #ecc in [0,1)
            cube[counter+5] = 2*math.pi*cube[counter+5] #omega in [0,2pi)
            counter += 6

    def mnest_loglike(self, cube, ndims, nparams):
        """
        Log likelihood function for MultiNest
        """
        return self.lnpost(cube)

    @property
    def mnest_analyzer(self): #This returns an Analyzer object! Not the stats
        """
        PyMultNest Analyzer object associated with fit
        """
        return pymultinest.Analyzer(self.n_params, self._mnest_basename)

    @property
    def evidence(self): #This returns evidence and evidence error from the Analyzer object
        """
        Log(evidence), Log(evidence error) from the MultiNest fit
        """
        s = self.mnest_analyzer.get_stats()
        return (s['global evidence'],s['global evidence error'])
        
    def __call__(self, p):
        #TransitModelInstance() returns the log posterior 
        return self.lnpost(p)

    def cost(self, p): #useful for scipy.optimize.minimize
        return -self.lnpost(p)
    
    def lnpost(self, par):
        #ln post = ln (prior*likelihood) = ln prior + ln likelihood
        
        #don't slice the cube! switch for Single and Binary models 
        if hasattr(self,'which'): p = [par[i] for i in range(9+6*self.lc.n_planets)]
        else: p = [par[i] for i in range(5+6*self.lc.n_planets)]
        
        prior = self.lnprior(p)
        if np.isfinite(prior): #if the prior is finite, then set likelihood
            like = self.lnlike(p)
        else:
            return prior #otherwise, just return the prior (aka -infinity)
        return prior + like #if prior is finite, return the ln posterior
                    
    def lnlike(self, p):
        try:
            flux_model = self.evaluate(p)
        except InvalidParameterError:
            return -np.inf
        #returns the normalised ln chi square statistic for the flux model (based on our input
        #parameters) vs the data that we got from kepler
        like_normalisation = np.log(1./(self.lc.flux_err*np.sqrt(2*np.pi))) #Gaussian normalisation
        return (like_normalisation + -0.5 * (flux_model - self.lc.flux)**2 / self.lc.flux_err**2).sum()
        
    def lnprior(self, p):
        flux_zp, rhostar, q1, q2, dilution = p[:5]

        #invalid parameter ranges
        if not (0 <= q1 <=1 and 0 <= q2 <= 1):
            return -np.inf
        if rhostar < 0:
            return -np.inf
        if not (0 <= dilution < 1):
            return -np.inf

        tot = 0

        # Apply stellar density prior if relevant.
        if self.lc.rhostar is not None:
            if self.lc.rhostar_pdf(rhostar) == 0.0: return -np.inf
            else: tot += np.log(self.lc.rhostar_pdf(rhostar))
            
        if self.lc.dilution is not None:
            if self.lc.dilution_pdf(dilution) == 0.0: return -np.inf
            else: tot += np.log(self.lc.dilution_pdf(dilution)) 

        for i in xrange(self.lc.n_planets):
            period, epoch, b, rprs, e, w = p[5+i*6:11+i*6]

            if not 0 < e < 1:
                return -np.inf
            if period <= 0:
                return -np.inf
            if rprs <= 0:
                return -np.inf
            if b < 0 or b > (1 + rprs):
                return -np.inf

            factor = 1.0
            if e > 0:
                factor = (1 + e * np.sin(w)) / (1 - e * e)

            aR = (rhostar * G * (period*DAY)**2 / (3*np.pi))**(1./3)
                
            arg = b * factor/aR
            if arg > 1.0:
                return -np.inf
                
            
            # Gaussian priors on period, epoch based on discovery measurements
            prior_p, prior_p_err = self.lc.planets[i]._period
            p_normalisation = np.log(1./(prior_p_err*np.sqrt(2*np.pi))) #Gaussian normalisation
            tot += -0.5*(period - prior_p)**2/prior_p_err**2 + p_normalisation

            prior_ep, prior_ep_err = self.lc.planets[i]._epoch
            e_normalisation = np.log(1./(prior_e_err*np.sqrt(2*np.pi))) #Gaussian normalisation
            tot += -0.5*(epoch - prior_ep)**2/prior_ep_err**2 + e_normalisation

            # normalised log-flat prior on rprs
            rprs_normalisation = np.log(0.3/0.005) #normalisation based on limits of prior range
            tot += np.log(1 / rprs * 1./rprs_normalisation)


            # Beta prior on eccentricity
            a,b = (0.4497, 1.7938)
            eccprior = 1/beta(a,b) * e**(a-1) * (1 - e)**(b-1)
            tot += np.log(eccprior)
            
        return tot

    def plot_planets(self, params, width=2, color='r', fig=None,
                     marker='o', ls='none', ms=0.5, **kwargs):
        
        if fig is None:
            fig = self.lc.plot_planets(width=width, **kwargs)

        # Scale widths for each plot by duration.
        maxdur = max([p.duration for p in self.lc.planets])
        widths = [width / (p.duration/maxdur) for p in self.lc.planets]

        depth = (1 - self.evaluate(params))*1e6
        
        for i,ax in enumerate(fig.axes):
            tfold = self.lc.t_folded(i) * 24
            close = self.lc.close(i, width=widths[i], only=True)
            ax.plot(tfold[close], depth[close], color=color, mec=color,
                    marker=marker, ls=ls, ms=ms, **kwargs)

        return fig

    @property
    def samples(self):
        if self.use_emcee:
            if not hasattr(self,'sampler') and self._samples is None:
                raise AttributeError('Must run MCMC (or load from file) '+
                                     'before accessing samples')
            
            if self._samples is not None:
                df = self._samples
            else:
                self._make_samples()
                df = self._samples

        else:
            if not(hasattr(self,'_mnest_basename')):
                raise AttributeError('Must run MultiNest before accessing samples')
            if not os.path.exists(self._mnest_basename + 'post_equal_weights.dat'):
                raise AttributeError('Must run MultiNest before accessing samples')
            if self._samples is not None:
                df = self._samples
            else:
                self._make_samples()
                df = self._samples
        return df
        
    def _make_samples(self):
        if self.use_emcee:
            flux_zp = self.sampler.flatchain[:,0]
            rho = self.sampler.flatchain[:,1]
            q1 = self.sampler.flatchain[:,2]
            q2 = self.sampler.flatchain[:,3]
            dilution = self.sampler.flatchain[:,4]

            df = pd.DataFrame(dict(flux_zp=flux_zp,
                                   rho=rho, q1=q1, q2=q2,
                                   dilution=dilution))

            for i in range(self.lc.n_planets):
                for j, par in enumerate(['period', 'epoch', 'b', 'rprs',
                                         'ecc', 'omega']):
                    column = self.sampler.flatchain[:, 5+j+i*6]
                    
                    if par=='omega':
                        column = column % (2*np.pi)
                        
                    df['{}_{}'.format(par,i+1)] = column
        else:
            post_samples = np.loadtxt(self._mnest_basename + 'post_equal_weights.dat')
            flux_zp = post_samples[:,0]
            rho = post_samples[:,1]
            q1 = post_samples[:,2]
            q2 = post_samples[:,3]
            dilution = post_samples[:,4]

            df = pd.DataFrame(dict(flux_zp=flux_zp,rho=rho,q1=q1,q2=q2,dilution=dilution))


            for i in range(self.lc.n_planets):
                for j,par in enumerate(['period', 'epoch', 'b', 'rprs',
                                         'ecc', 'omega']):
                    column = post_samples[:,5+j+i*6]

                    if par == 'omega':
                        column = column %(2*np.pi)
                    df['{}_{}'.format(par,i+1)] = column


        self._samples = df

    def triangle(self, params=None, i=0, query=None, extent=0.999,
                 planet_only=False, passedfrombinary=False,
                 **kwargs):
        """
        Makes a nifty corner plot for planet i

        Uses :func:`triangle.corner`.

        :param params: (optional)
            Names of columns to plot.  Set planet_only to be ``True``
            to leave out star params.

        :param i:
            Planet number (starting from 0)

        :param query: (optional)
            Optional query on samples.

        :param extent: (optional)
            Will be appropriately passed to :func:`triangle.corner`.

        :param **kwargs:
            Additional keyword arguments passed to :func:`triangle.corner`.

        :return:
            Figure oject containing corner plot.
            
        """
        if triangle is None:
            raise ImportError('please run "pip install triangle_plot".')
        
        if params is None:
            params = []
            if not(planet_only):
                params.append('dilution')
                params.append('rho')
                params.append('q1')
                params.append('q2')
            for par in ['period', 'epoch', 'b', 'rprs',
                        'ecc', 'omega']:
                params.append('{}_{}'.format(par, i+1))
        elif passedfrombinary:
            for par in ['period', 'epoch', 'b', 'rprs',
                        'ecc', 'omega']:
                params.append('{}_{}'.format(par, i+1))


        df = self.samples

        if query is not None:
            df = df.query(query)

        #convert extent to ranges, but making sure
        # that truths are in range.
        extents = []
        remove = []
        for i,par in enumerate(params):
            values = df[par]
            qs = np.array([0.5 - 0.5*extent, 0.5 + 0.5*extent])
            minval, maxval = values.quantile(qs)
            if 'truths' in kwargs:
                datarange = maxval - minval
                if kwargs['truths'][i] < minval:
                    minval = kwargs['truths'][i] - 0.05*datarange
                if kwargs['truths'][i] > maxval:
                    maxval = kwargs['truths'][i] + 0.05*datarange
            extents.append((minval,maxval))
            
        return triangle.corner(df[params], labels=params, 
                               extents=extents, **kwargs)

    def save_hdf(self, filename, path='', overwrite=False, append=False):
        """Saves object data to HDF file (only works if MCMC is run)

        Samples are saved to /samples location under given path,
        and object properties are also attached, so suitable for
        re-loading via :func:`TransitModel.load_hdf`.
        
        :param filename:
            Name of file to save to.  Should be .h5 file.

        :param path: (optional)
            Path within HDF file structure to save to.

        :param overwrite: (optional)
            If ``True``, delete any existing file by the same name
            before writing.

        :param append: (optional)
            If ``True``, then if a file exists, then just the path
            within the file will be updated.
        """
        
        if os.path.exists(filename):
            store = pd.HDFStore(filename)
            if path in store:
                store.close()
                if overwrite:
                    os.remove(filename)
                elif not append:
                    raise IOError('{} in {} exists.  Set either overwrite or append option.'.format(path,filename))
            else:
                store.close()

                
        self.samples.to_hdf(filename, '{}/samples'.format(path))

        store = pd.HDFStore(filename)
        attrs = store.get_storer('{}/samples'.format(path)).attrs
        attrs.width = self.width
        attrs.continuum_method = self.continuum_method
        attrs._mnest_basename = self._mnest_basename
        attrs.koinum = self.koinum
        if hasattr(self,'which'): attrs.which = self.which
        attrs.lc_type = type(self.lc)
        
        store.close()

        self.lc.save_hdf(filename, path=path, append=True)
        
    @classmethod
    def load_hdf(cls, filename, path=''):
        """
        A class method to load a saved StarModel from an HDF5 file.

        File must have been created by a call to :func:`StarModel.save_hdf`.

        :param filename:
            H5 file to load.

        :param path: (optional)
            Path within HDF file.

        :return:
            :class:`StarModel` object.
        """
        store = pd.HDFStore(filename)
        try:
            samples = store['{}/samples'.format(path)]
            attrs = store.get_storer('{}/samples'.format(path)).attrs        
        except:
            store.close()
            raise
        width = attrs.width
        continuum_method = attrs.continuum_method
        koinum = attrs.koinum
        _mnest_basename = attrs._mnest_basename
        lc_type = attrs.lc_type
        store.close()

        lc = lc_type.load_hdf(filename, path=path)

        mod = cls(lc, width=width, continuum_method=continuum_method)
        mod._samples = samples
        
        return mod
    

class BinaryTransitModel(TransitModel):
    """
    TransitModel of two stars, for now can only fit with MultiNest

    :param lc:
        BinaryLightCurve object.

    :param which:
        e.g., for three-planet system: ['A', 'A', 'B']
        which defaults to A for all
        if ``len(which) < n_planets`` then defaults missing values to A

    """
    def __init__(self, lc, which=None,width = 2,**kwargs):
        
        if which == None:
            self.which = ['A'] * lc.n_planets
        elif len(which) < lc.n_planets:
            while len(which) < lc.n_planets:
                which.append('A')
            self.which = which
        else: self.which = which

        super(BinaryTransitModel,self).__init__(lc,width = width,**kwargs)

    def evaluate(self, par):
        """
        Evaluates light curve model at light curve times.

        Difference with TransitModel is that there are now two stars.

        :param p:
            Parameter vector, of length 1 + 4*2 + 6*Nplanets
            p[0] = flux zero-point
            p[1:5] = [rhostarA, q1A, q2A, dilutionA]
            p[5:9] = [rhostarB, q1B, q2B, dilutionB]
            p[9+i*6 : 15+i*6] = [per, ep, b, rprs, e, w] for i-th planet  
        """
        
        # So as to be careful to not pass slices of par around...
        p = [par[i] for i in range(9+6*self.lc.n_planets)]
        
        pA,pB = p[0:5],[p[0],p[5],p[6],p[7],p[8]]

        f = self.continuum(p[0], self.lc.t)
        fA = np.copy(f)
        fB = np.copy(f)

        close_A = np.zeros_like(self.lc.t).astype(bool)
        close_B = np.zeros_like(self.lc.t).astype(bool)

        # Must use self.which to determine which star parameters
        # get passed to lc_eval for each planet.
        # Build close_A and close_B properly, depending on self.which
        for i in xrange(self.lc.n_planets):
            if self.which[i] == 'A':
                pA = np.append(pA,p[9+i*6 : 15+i*6])
                close_A += self.lc.close(i,width=self.width)
            else:
                pB = np.append(pB,p[9+i*6 : 15+i*6])
                close_B += self.lc.close(i,width=self.width)

        fA[close_A] = (lc_eval(pA[1:],self.lc.t[close_A],texp=self.lc.texp))
        fB[close_B] = (lc_eval(pB[1:],self.lc.t[close_B],texp=self.lc.texp))
        depthA = 1-fA
        depthB = 1-fB
        totaldepth = depthA + depthB
        f = 1 - totaldepth
        
        return f

    def mnest_prior(self, cube, ndims, nparams):
        """
        Transforms the unit cube into parameter cube

        Uses simple flat priors, more complicated transformations will occur in lnprior

        Priors are flux_zp, [rhostar, q1, q2, dilution] for each star
        and [period, epoch, b, rprs, ecc, omega] for each planet
        """
        #flat priors
        cube[0] = 0.04*cube[0] + 0.98 #flux_zp in [0.98,1.02)

        #stellar parameters
        cube[1] = 199.999*cube[1] + 1e-4 #rhostarA in [1e-4, 200)
        # cube[2] = unchanged # q1A in [0,1)
        # cube[3] = unchanged # q2A in [0,1)
        # cube[4] = unchanged # dilutionA in [0,1)

        cube[5] = 199.999*cube[5] + 1e-4 #rhostarB in [1e-4,200)
        #cube[6] = unchanged # q1B in [0,1)
        #cube[7] = unchanged # q2B in [0,1)
        #cube[8] = unchanged #dilutionB in [0,1)

        counter = 9
        for i in xrange(self.lc.n_planets): #iterating over each planet in the light curve system
            #grabbing these prior values as the mean and error for flat priors
            prior_p_mu, prior_p_sig = self.lc.planets[i]._period
            prior_ep_mu, prior_ep_sig = self.lc.planets[i]._epoch
            #setting the flat priors between mu +- 10sigma for period, epoch
            cube[counter] = 20*prior_p_sig*cube[counter] + prior_p_mu - 10*prior_p_sig #period
            cube[counter+1] = 20*prior_ep_sig*cube[counter+1] + prior_ep_mu - 10*prior_ep_sig #epoch
            cube[counter+2] = 2*cube[counter+2]#b in [0,2)
            cube[counter+3] = 0.499*cube[counter+3] + 0.001#rprs in [0.001,0.5)
            # cube[counter+4] = unchanged #ecc in [0,1)
            cube[counter+5] = 2*math.pi*cube[counter+5] #omega in [0,2pi)
            counter += 6

    def lnprior(self, p):
        flux_zp, rhostarA, q1A, q2A, dilutionA = p[:5]
        rhostarB, q1B, q2B, dilutionB = p[5:9]

        #invalid parameter ranges
        if not (0 <= q1A <=1 and 0 <= q2A <= 1):
            return -np.inf
        if rhostarA < 0:
            return -np.inf
        if not (0 <= dilutionA < 1):
            return -np.inf

        if not (0 <= q1B <=1 and 0 <= q2B <= 1):
            return -np.inf
        if rhostarB < 0:
            return -np.inf
        if not (0 <= dilutionB < 1):
            return -np.inf

        tot = 0

        # Apply stellar density prior if relevant.
        if self.lc.rhostarA is not None:
            if self.lc.rhostarA_pdf(rhostarA) == 0.0: return -np.inf
            else: tot += np.log(self.lc.rhostarA_pdf(rhostarA))
        
        if self.lc.rhostarB is not None:
            if self.lc.rhostarB_pdf(rhostarB) == 0.0: return -np.inf
            else: tot += np.log(self.lc.rhostarB_pdf(rhostarB)) 
       
        # Apply dilution prior if relevant
        if self.lc.dilutionA is not None:
            if self.lc.dilutionA_pdf(dilutionA) == 0.0: return -np.inf
            else: tot += np.log(self.lc.dilutionA_pdf(dilutionA))
       
        if self.lc.dilutionB is not None:
            if self.lc.dilutionB_pdf(dilutionB) == 0.0: return -np.inf
            else: tot += np.log(self.lc.dilutionB_pdf(dilutionB))
        
        for i in xrange(self.lc.n_planets):
            period, epoch, b, rprs, e, w = p[9+i*6:11+i*6]

            #more invalid parameter ranges
            if not 0 < e < 1:
                return -np.inf
            if period <= 0:
                return -np.inf
            if rprs <= 0:
                return -np.inf
            if b < 0 or b > (1 + rprs):
                return -np.inf

            factor = 1.0
            if e > 0:
                factor = (1 + e * np.sin(w)) / (1 - e * e)

            aR = (rhostar * G * (period*DAY)**2 / (3*np.pi))**(1./3)
                
            arg = b * factor/aR
            if arg > 1.0:
                return -np.inf
                
            
            # Gaussian priors on period, epoch based on discovery measurements
            prior_p, prior_p_err = self.lc.planets[i]._period
            p_normalisation = np.log(1./(prior_p_err*np.sqrt(2*np.pi))) #Gaussian normalisation
            tot += -0.5*(period - prior_p)**2/prior_p_err**2 + p_normalisation

            prior_ep, prior_ep_err = self.lc.planets[i]._epoch
            e_normalisation = np.log(1./(prior_e_err*np.sqrt(2*np.pi))) #Gaussian normalisation
            tot += -0.5*(epoch - prior_ep)**2/prior_ep_err**2 + e_normalisation

            # log-flat prior on rprs
            rprs_normalisation = np.log(0.3/0.005) #normalisation based on limits of prior range
            tot += np.log(1/rprs * normalisation)


            # Beta prior on eccentricity
            a,b = (0.4497, 1.7938)
            eccprior = 1/beta(a,b) * e**(a-1) * (1 - e)**(b-1)
            tot += np.log(eccprior)
            
        return tot

    def _make_samples(self):
        post_samples = np.loadtxt(self._mnest_basename + 'post_equal_weights.dat')
        flux_zp = post_samples[:,0]
        rhoA = post_samples[:,1]
        q1A = post_samples[:,2]
        q2A = post_samples[:,3]
        dilutionA = post_samples[:,4]
        rhoB = post_samples[:,5]
        q1B = post_samples[:,6]
        q2B = post_samples[:,7]
        dilutionB = post_samples[:,8]

        df = pd.DataFrame(dict(flux_zp=flux_zp,rhoA=rhoA,q1A=q1A,q2A=q2A,dilutionA=dilutionA,
                                rhoB=rhoB,q1B=q1B,q2B=q2B,dilutionB=dilutionB))


        for i in range(self.lc.n_planets):
            whichplanet = self.which[i]
            for j,par in enumerate(['period', 'epoch', 'b', 'rprs',
                                     'ecc', 'omega']):
                column = post_samples[:,9+j+i*6]

                if par == 'omega':
                    column = column %(2*np.pi)
                df['{}_{}_{}'.format(par,i+1,whichplanet)] = column


        self._samples = df

    def triangle(self, params=None, planet_only=False,**kwargs):
        if triangle is None:
            raise ImportError('please run "pip install triangle_plot".')

        if params is None:
            if planet_only:
                params = []
            else: params = ['dilutionA','rhostarA','q1A','q2A','dilutionB','rhostarB','q1B','q2B']
        
        super(BinaryTransitModel, self).triangle(params,planet_only=planet_only,passedfrombinary=True,**kwargs)


    @classmethod
    def load_hdf(cls, filename, path=''):
        """
        A class method to load a saved BinaryTransitModel from an HDF5 file.

        File must have been created by a call to :func:`StarModel.save_hdf`.

        :param filename:
            H5 file to load.

        :param path: (optional)
            Path within HDF file.

        :return:
            :class:`BinaryTransitModel` object.
        """
        store = pd.HDFStore(filename)
        try:
            samples = store['{}/samples'.format(path)]
            attrs = store.get_storer('{}/samples'.format(path)).attrs        
        except:
            store.close()
            raise
        which = attrs.which
        width = attrs.width
        continuum_method = attrs.continuum_method
        lc_type = attrs.lc_type
        koinum = attrs.koinum
        _mnest_basename = attrs._mnest_basename
        store.close()

        lc = lc_type.load_hdf(filename, path=path)

        mod = cls(lc, which=which,width=width, continuum_method=continuum_method)
        mod._samples = samples
        
        return mod
