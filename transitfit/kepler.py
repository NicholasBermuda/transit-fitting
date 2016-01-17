from __future__ import print_function, division

import re
import pandas as pd
import numpy as np

import kplr 

from .lightcurve import LightCurve, Planet, BinaryLightCurve

KEPLER_CADENCE = 1626./86400

def lc_dataframe(lc):
    """Returns a pandas DataFrame of given lightcurve data
    """
    with lc.open() as f:
        data = f[1].data

    data = np.array(data).byteswap().newbyteorder()

    return pd.DataFrame(data)    

def all_LCdata(koi, mask_bad=False):
    """
    Returns all data for a given koi, in a pandas dataframe

    PDCSAP_FLUX is quarter-stitched and normalized, with bad points masked 
    """
    df = pd.DataFrame()
    for lc in koi.get_light_curves():
        if re.search('_llc\.fits', lc.filename):
            newdf = lc_dataframe(lc)
            normalfactor = newdf['PDCSAP_FLUX'].mean()
            newdf['PDCSAP_FLUX'] /= normalfactor
            newdf['PDCSAP_FLUX_ERR'] /= normalfactor
            df = pd.concat([df, newdf])
            
    if mask_bad:
        ok = np.isfinite(df['PDCSAP_FLUX']) & (df['SAP_QUALITY']==0)
    else:
        ok = np.ones(len(df)).astype(bool)
    return df[ok]

def kepler_planets(koinum, i):
    #reads in the planets from a koi and adds them to the list of planets
    #as a Planet object
    client = kplr.API()
    
    if type(i)==int:
        ilist = [i]
    else:
        ilist = i

    koi_list = [koinum + i*0.01 for i in ilist]

    planets = []
    kois = []
    for k in koi_list:
        k = client.koi(k)
        planets.append(Planet((k.koi_period, k.koi_period_err1),
                              (k.koi_time0bk, k.koi_time0bk_err1),
                              k.koi_duration/24,
                              name=k.kepoi_name))
        kois.append(k)

    return kois, planets
    

class KeplerLightCurve(LightCurve):
    """A LightCurve of a Kepler star

    :param koinum:
        KOI number (integer).

    :param i:
        Planet number, either integer (1 through koi_count),
        list of integers, or None (in which case all planets will be modeled).
        
    """
    def __init__(self, koinum, i=None,**kwargs):
        self.koinum = koinum #used for multinest basename folder organisation
        client = kplr.API() #interacting with Kepler archive
        koi = client.koi(koinum + 0.01) #getting the first planet to download info
        if i is None: #if there is no input
            i = range(1,koi.koi_count+1) #then we create an array of all the planets
        lcdata = all_LCdata(koi) #downloads all the light curve data

        #mask out NaNs
        mask = ~np.isfinite(lcdata['PDCSAP_FLUX']) | lcdata['SAP_QUALITY']

        kois, planets = kepler_planets(koinum, i=i) #get the kois and planets
        self.kois = kois
        
        super(KeplerLightCurve, self).__init__(lcdata['TIME'],
                                                 lcdata['PDCSAP_FLUX'],
                                                 lcdata['PDCSAP_FLUX_ERR'],
                                                 mask=mask, planets=planets,
                                                 texp=KEPLER_CADENCE, **kwargs)

    @property
    def archive_params(self): 
    #reads in the parameters from the archive
        params = [1, self.kois[0].koi_srho, 0.5, 0.5, 0]
        
        for k in self.kois:
            params += [k.koi_period, k.koi_time0bk, k.koi_impact, k.koi_ror, 0, 0]

        return params

    def archive_light_curve(self, t):
    #reads in the light curve data from the archive
        return self.light_curve(self.archive_params, t)

class BinaryKeplerLightCurve(BinaryLightCurve):
    """BinaryLightCurve of a Kepler star

    :param koinum:
        KOI number (integer)

    :param i:
        Planet number, either integer (1 through koi_count),
        list of integers, or None (in which case all planets will be modeled).
    """ 
    def __init__(self, koinum, i=None,rhostarA=None,rhostarB=None,dilution = None,**kwargs):
        self.koinum = koinum #used for multinest basename folder organisation
        client = kplr.API() #interacting with Kepler archive
        koi = client.koi(koinum + 0.01) #getting the first planet to download info
        if i is None: #if there is no input
            i = range(1,koi.koi_count+1) #then we create an array of all the planets
        lcdata = all_LCdata(koi) #downloads all the light curve data

        #mask out NaNs
        mask = ~np.isfinite(lcdata['PDCSAP_FLUX']) | lcdata['SAP_QUALITY']

        kois, planets = kepler_planets(koinum, i=i) #get the kois and planets
        self.kois = kois
        
        super(BinaryKeplerLightCurve, self).__init__(lcdata['TIME'],
                                                 lcdata['PDCSAP_FLUX'],
                                                 lcdata['PDCSAP_FLUX_ERR'],
                                                 rhostarA=rhostarA,rhostarB=rhostarB,
                                                 dilution = dilution,
                                                 mask=mask, planets=planets,
                                                 texp=KEPLER_CADENCE, **kwargs)        


#    @classmethod
#    def from_hdf(cls, *args, **kwargs):
#        raise NotImplementedError
    
#    @classmethod
#    def from_df(cls, df, **kwargs):
#        raise NotImplementedError
