def prepare_mainsubstituter():
    """Prepare and return a |Substituter| object for the main `__init__`
    file of *HydPy*."""
    substituter = Substituter()
    for module in (builtins, numpy, datetime, unittest, doctest, inspect, io,
                   os, sys, time, collections, itertools, subprocess, scipy,
                   typing):
        substituter.add_module(module)
    for subpackage in (auxs, core, cythons, exe):
        for dummy, name, dummy in pkgutil.walk_packages(subpackage.__path__):
            full_name = subpackage.__name__ + '.' + name
            substituter.add_module(importlib.import_module(full_name))
    substituter.add_modules(models)
    for cymodule in (annutils, smoothutils, pointerutils):
        substituter.add_module(cymodule, cython=True)
    substituter._short2long['|pub|'] = ':mod:`~hydpy.pub`'
    substituter._short2long['|config|'] = ':mod:`~hydpy.config`'
    return substituter