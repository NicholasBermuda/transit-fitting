__version__ = '0.1'

try:
    __TRANSITFIT_SETUP__
except NameError:
    __TRANSITFIT_SETUP__ = False

if not __TRANSITFIT_SETUP__:
    from .lightcurve import LightCurve, BinaryLightCurve
    from .kepler import KeplerLightCurve, BinaryKeplerLightCurve
    from .fitter import TransitModel, BinaryTransitModel
