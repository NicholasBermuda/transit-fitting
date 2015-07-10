from __future__ import print_function, division

import numpy as np
from numpy import ma
import pandas as pd

import os, os.path

import matplotlib.pyplot as plt

from transit import Central, System, Body

from .utils import t_folded, lc_eval
#t_folded returns the folded time using t, period, epoch
#lc_eval returns numpy ndarray of flux values of a transit model



class Planet(object):
    """
    Initialize with period, epoch as 2-element tuples, 
    """
    def __init__(self, period, epoch, duration, name=''):

        if type(period) in (float, int):
            period = (period, np.nan)
        if type(epoch) in (float, int):
            epoch = (epoch, np.nan)

        assert len(period)==2
        assert len(epoch)==2
        
        self._period = tuple(period)
        self._epoch = tuple(epoch)

        self.duration = duration

        self.name = name

    @property
    def period(self):
        return self._period[0]

    @property
    def epoch(self):
        return self._epoch[0]

    @property
    def e_period(self):
        return self._period[1]

    @property
    def e_epoch(self):
        return self._epoch[1]
        
    def t_folded(self, t):
        return t_folded(t, self.period, self.epoch)

    def close(self, t, width=2):
        return np.absolute(self.t_folded(t)) < width*self.duration

    def in_transit(self, t, width=0.55):
        return self.close(t, width=width)

    def ith_transit(self, t, i, width=2):
        """Returns True around ith transit (as measured from epoch)
        """
        per, ep = (self.period, self.epoch)
        close = np.absolute(((t - ep + per/2) / per) 
                           - per/2 - i ) < width*self.duration
        return close
    
    
class LightCurve(object):
    """Object holding time/flux data and info about transiting planets

    :param time,flux,flux_err:
        Time series data.

    :param texp:
        Exposure time.  If not provided, will be assumed to be median
        of delta-t.
        
    """
    def __init__(self, time, flux, flux_err=0.0001,
                 mask=None, texp=None, planets=None,
                 detrend=True):

        
        if mask is None:
            mask = ~np.isfinite(flux)
        self.mask = np.array(mask).astype(bool)

        if texp is None:
            texp = np.median(time[1:]-time[:-1])
        self.texp = texp
        
        if planets is None:
            planets = []
        self.planets = planets

        self._time = np.array(time)
        self._flux = np.array(flux)
        self._flux_err = np.array(flux_err)

        if detrend:
            self.median_detrend()
        else:
            self._detrended_flux = np.array(flux)

    @property
    def t(self):
        return self.time

    @property
    def f(self):
        return self.f
    
    @property
    def time(self):
        return self._time[~self.mask]
    
    @property
    def rawflux(self):
        return self._flux[~self.mask]
        
    @property
    def flux_err(self):
        return self._flux_err[~self.mask]
        
    @property
    def flux(self):
        return self._detrended_flux[~self.mask]

    def median_detrend(self, window=75): 
        #rolling median to normalise the flux read from archive
        f = self._flux.copy()
        f[self.any_intransit] = np.nan
        f_median = pd.rolling_median(f, 75, center=True,
                                     min_periods=1)
        self._detrended_flux = self._flux / f_median

    @property
    def n_planets(self):
        return len(self.planets)
        
    def add_planet(self, planet):
        self.planets.append(planet)

    def t_folded(self, i=0):
        """Times folded on the period and epoch of planet i
        """
        return self.planets[i].t_folded(self.time)

    def close(self, i=0, width=2, only=False):
        """Boolean array with True everywhere within width*duration of planet i

        if only, then any cadences with other planets also get masked out
        """
        close = self.planets[i].close(self.time, width=width)
        if only:
            for j in range(self.n_planets):
                if j==i:
                    continue
                close &= ~self.close(j, width=width)
            
        return close

    @property
    def anyclose(self):
        close = np.zeros_like(self.time).astype(bool)
        for i in range(self.n_planets):
            close += self.close(i)
        return close

    def intransit(self, i=0, width=0.55):
        """Boolean mask True everywhere within 0.6*duration of planet i
        """
        return self.planets[i].in_transit(self.time, width=width)

    @property
    def any_intransit(self):
        intrans = np.zeros_like(self.time).astype(bool)
        for i in range(self.n_planets):
            intrans += self.intransit(i)
        return intrans

    @property
    def n_transits(self):
        tspan = self.time[-1] - self.time[0]
        return [(tspan // p.period) + 1 for p in self.planets]
        
    def ith_transit(self, i, i_planet=0, width=2):
        """returns True around i-th transit for planet number "i_planet"
        """
        return self.planets[i_planet].ith_transit(self.t, i, width=width)

    def transit_stack(self, i=0, width=2):
        """returns a 2-d array of times/fluxes with subsequent transits in each row
        """

    @property
    def default_params(self):
        """Quick and dirty guesses for params

        """
        params = [1, 4, 0.5, 0.5, 0]


        for i,p in enumerate(self.planets):
            minflux = np.median(self.flux[self.close(i, width=0.2, only=True)])
            ror = np.sqrt(1 - minflux)
            params += [p.period, p.epoch, 0.5, ror, 0, 0]

        return params
        
    def plot_planets(self, width=2, **kwargs):
        n = self.n_planets #number of planets
        fig, axs = plt.subplots(n, 1, sharex=True) #setting up the number of subplots

        fig.set_figwidth(8) #dimensions of subplots
        fig.set_figheight(2*n)
        
        # Scale widths for each plot by duration.
        maxdur = max([p.duration for p in self.planets])
        widths = [width / (p.duration/maxdur) for p in self.planets]
        
        for i,ax in enumerate(axs): #plotting each individual planet with plot_planet
        #breaks if there is only one planet
            self.plot_planet(i, ax=ax, width=widths[i], **kwargs)
            ax.set_xlabel('')
            ax.set_ylabel('')
            yticks = ax.get_yticks()
            ax.set_yticks(yticks[1:])

        axs[n//2].set_ylabel('Depth [ppm]', fontsize=18)
        axs[-1].set_xlabel('Hours from mid-transit', fontsize=18)

        fig.subplots_adjust(hspace=0)
            
        return fig
        
        
    def plot_planet(self, i=0, width=2, ax=None,
                    marker='o', ls='none', color='k',
                    ms=0.3, alpha=1, 
                    **kwargs):
        """Plots planet i; masking out others, if present
        """
        if ax is None: #setting the axes
            fig, ax = plt.subplots(1, 1)
        else:
            fig = plt.gcf()

        tfold = self.t_folded(i) * 24 #folding the time over the period
        close = self.close(i, width=width, only=True) #look at times close to transit
        depth = (1 - self.flux)*1e6 #putting transit into ppm
        
        ax.plot(tfold[close], depth[close], color=color, #plotting the transit
                marker=marker, ms=ms, ls=ls, alpha=alpha, **kwargs)

        ax.invert_yaxis() #labelling axes
        ax.set_xlabel('Time from mid-transit (hours)', fontsize=18)
        ax.set_ylabel('Depth (ppm)', fontsize=18)

        ax.annotate(self.planets[i].name, xy=(0.8,0.05), #planet name/KOI number
                    xycoords='axes fraction', fontsize=14)

        ax.annotate('P = {}d'.format(self.planets[i].period), #period in days
                    xy=(0.05, 0.05), xycoords='axes fraction',
                    fontsize=14)
        
        return fig

    def save_hdf(self, filename, path='', overwrite=False, append=False):
        """Saves object data to HDF file

        Suitable for re-loading via :func:`LightCurve.load_hdf`.
        
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

        self.dataframe.to_hdf(filename, '{}/lc'.format(path))

        store = pd.HDFStore(filename)
        attrs = store.get_storer('{}/lc'.format(path)).attrs
        attrs.texp = self.texp
        attrs.planets = self.planets
        store.close()

    @classmethod
    def load_hdf(cls, filename, path=''):
        """
        A class method to load a saved LightCurve from an HDF5 file.

        File must have been created by a call to :func:`LightCurve.save_hdf`.

        :param filename:
            H5 file to load.

        :param path: (optional)
            Path within HDF file.

        :return:
            :class:`LightCurve` object.
        """
        store = pd.HDFStore(filename)
        try:
            df = store['{}/lc'.format(path)]
            attrs = store.get_storer('{}/lc'.format(path)).attrs        
        except:
            store.close()
            raise
        texp = attrs.texp
        planets = attrs.planets
        store.close()

        return cls.from_df(df, texp=texp, planets=planets)
    
    @property
    def dataframe(self):
        """
        Return data as a pandas DataFrame
        """
        df = pd.DataFrame()
        df['time'] = self._time
        df['flux'] = self._flux
        df['flux_err'] = self._flux_err
        df['mask'] = self.mask
        df['detrended_flux'] = self._detrended_flux

        return df
            
    @classmethod
    def from_df(cls, df, **kwargs):
        new = cls(df['time'], df['flux'], df['flux_err'],
                  mask=df['mask'], detrend=False,
                  **kwargs)

        new._detrended_flux = df['detrended_flux']
        
        return new
