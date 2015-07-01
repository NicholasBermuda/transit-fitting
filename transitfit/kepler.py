from __future__ import print_function, division

import re
import pandas as pd
import numpy as np

import kplr

from .transitmodel import TransitModel

KEPLER_CADENCE = 1626./86400

def lc_dataframe(lc):
    """Returns a pandas DataFrame of given lightcurve data
    """
    with lc.open() as f:
        data = f[1].data

    data = np.array(data).byteswap().newbyteorder()

    return pd.DataFrame(data)    

def all_LCdata(koi, mask_bad=True):
    """
    Returns all data for a given koi, in a pandas dataframe

    PDCSAP_FLUX is quarter-stitched and normalized, with bad points masked 
    """
    df = pd.DataFrame()
    for lc in koi.get_light_curves():
        if re.search('_llc\.fits', lc.filename):
            newdf = lc_dataframe(lc)
            newdf['PDCSAP_FLUX'] /= newdf['PDCSAP_FLUX'].mean()
            df = pd.concat([df, newdf])
            
    if mask_bad:
        ok = np.isfinite(df['PDCSAP_FLUX']) & (df['SAP_QUALITY']==0)
    return df[ok]

class KeplerTransitModel(TransitModel):
    """A TransitModel of a Kepler star

    :param koinum:
        KOI number (integer).

    :param i:
        Planet number, either integer (1 through koi_count),
        list of integers, or None (in which case all planets will be modeled).
        
    """
    def __init__(self, koinum, i=None):
        client = kplr.API()
        koi = client.koi(koinum + 0.01)
        count = koi.koi_count
        lcdata = all_LCdata(koi)

        if type(i)==int:
            ilist = [i]
        elif i is None:
            ilist = range(1, count+1)
        else:
            ilist = i
            
        koi_list = [koinum + i*0.01 for i in ilist]

        periods = []
        epochs = []
        durations = []
        kois = []
        for k in koi_list:
            k = client.koi(k)
            periods.append(k.koi_period)
            epochs.append(k.koi_time0bk)
            durations.append(k.koi_duration/24) # in days
            kois.append(k)

        self.kois = kois
        
        super(KeplerTransitModel, self).__init__(lcdata['TIME'],
                                                 lcdata['PDCSAP_FLUX'],
                                                 lcdata['PDCSAP_FLUX_ERR'],
                                                 period=periods, epoch=epochs,
                                                 duration=durations, texp=KEPLER_CADENCE)

    @property
    def archive_params(self):
        params = [self.kois[0].koi_srho, 0.5, 0.5, 0]
        
        for k in self.kois:
            params += [k.koi_period, k.koi_time0bk, k.koi_impact, k.koi_ror, 0, 0]

        return params

    def archive_light_curve(self, t):
        return self.light_curve(self.archive_params, t)
