#!/usr/bin/env python
"""
A command-line program to fit a BinaryTransitModel using the transitfit package

Input argument is name of a folder that contains a file
called ``model.ini``, which is a config file containing the
information needed to initialise a BinaryTransitModel object;
``koinum``, the list of planets you want to fit, ``which``
and the name of the star model you want to import from
the isochrones package
"""

import argparse

if __name__=='__main__':
