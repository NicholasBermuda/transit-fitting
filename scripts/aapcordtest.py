#!/usr/bin/env python
from transitfit import LightCurve, KeplerLightCurve, TransitModel

lc = KeplerLightCurve(1422,1)

model = TransitModel(lc,which=which,light_curve='transit')
model.fit_polychord(basename=('polychains/1-'))