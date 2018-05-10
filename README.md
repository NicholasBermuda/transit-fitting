# transitfit

This package harnesses [kplr](https://github.com/dfm/kplr) and [transit](https://github.com/dfm/transit) in the service of Pythonic transit modeling made super-easy (and super-duper-easy for *Kepler* data).  Check out the [demo notebook](https://github.com/timothydmorton/transit-fitting/blob/master/notebooks/demo.ipynb) for a quick look.  

This branch of the repository is used as part of Nicholas Barton's Senior Thesis completed in partial requirement for his degree at Princeton University. It extends Tim Morton's work for use with binary star systems with multiple planets, and has implementations of both MultiNest and PolyChord for Bayesian evidence calculations. When combined with the appropriate initialisation files, it can be used to determine the most likely distribution of planets around the two stellar components of a binary star.

This branch also allows users to evaluate light curves with [batman](https://github.com/lkreidberg/batman).

### Note that this work is not actively maintained. Use at your own risk!

[n.b., at present, this will only work with [Tim Morton's fork of `transit`](https://github.com/timothydmorton/transit).]

Installation
----------

    git clone https://github.com/NicholasBermuda/transit-fitting
    cd transit-fitting
    python setup.py install

Attribution
----------

See [main repo](https://github.com/timothydmorton/transit).

