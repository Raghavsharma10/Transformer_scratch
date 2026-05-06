def get_optimizer_impstr(optimizer_name):
    """
    Returns the import string for the optimizer
    """
    possibilities = {"bfgs":"BFGS",
                     "bfgslinesearch":"BFGSLineSearch",
                     "fire":"FIRE",
                     "goodoldquasinewton":"GoodOldQuasiNewton",
                     "hesslbfgs":"HessLBFGS",
                     "lbfgs":"LBFGS",
                     "lbfgslinesearch":"LBFGSLineSearch",
                     "linelbfgs":"LineLBFGS",
                     "mdmin":"MDMin",
                     "ndpoly":"NDPoly",
                     "quasinewton":"QuasiNewton",
                     "scipyfmin":"SciPyFmin",
                     "scipyfminbfgs":"SciPyFminBFGS",
                     "scipyfmincg":"SciPyFminCG",
                     "scipyfminpowell":"SciPyFminPowell",
                     "scipygradientlessoptimizer":"SciPyGradientlessOptimizer",
                     }
    
    current_val = possibilities.get(optimizer_name.lower())
    
    if current_val:
        return "from ase.optimize import {} as custom_optimizer".format(current_val)
    else:
        package,current_val = optimizer_name.rsplit('.',1)
        return "from ase.optimize.{} import {} as custom_optimizer".format(package,current_val)