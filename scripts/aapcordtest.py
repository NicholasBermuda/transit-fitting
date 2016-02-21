#!/usr/bin/env python
from transitfit import LightCurve, KeplerLightCurve, TransitModel, BinaryLightCurve, BinaryKeplerLightCurve, BinaryTransitModel

lc = BinaryLightCurve.load_hdf('fits/K01422/AAAAA/1422_lc.h5')

model = BinaryTransitModel(lc,which=which,light_curve='transit')
model.fit_polychord(basename=('fits/K01422/AAAAA/polychains/1-'))