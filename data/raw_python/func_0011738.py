def newmodel(f, G, y0, name='NewModel', modelType=ItoModel):
    """Use the functions f and G to define a new Model class for simulations. 

    It will take functions f and G from global scope and make a new Model class
    out of them. It will automatically gather any globals used in the definition
    of f and G and turn them into attributes of the new Model.

    Args:
      f: callable(y, t) (defined in global scope) returning (n,) array
         Scalar or vector-valued function to define the deterministic part
      G: callable(y, t) (defined in global scope) returning (n,m) array
         Optional scalar or matrix-valued function to define noise coefficients
         of a stochastic system. This should be ``None`` for an ODE system.
      y0 (Number or array): Initial condition
      name (str): Optional class name for the new model
      modelType (type): The type of model to simulate. Must be a subclass of
        nsim.Model, for example nsim.ODEModel, nsim.ItoModel or 
        nsim.StratonovichModel. The default is nsim.ItoModel.

    Returns: 
      new class (subclass of Model)

    Raises:
      SimValueError, SimTypeError
    """
    if not issubclass(modelType, Model):
        raise SimTypeError('modelType must be a subclass of nsim.Model')
    if not callable(f) or (G is not None and not callable(G)):
        raise SimTypeError('f and G must be functions of y and t.')
    if G is not None and f.__globals__ is not G.__globals__:
        raise SimValueError('f and G must be defined in the same place')
    # TODO: validate that f and G are defined at global scope.
    # TODO: Handle nonlocals used in f,G so that we can lift this restriction.
    if modelType is ODEModel and G is not None and not np.all(G == 0.0):
        raise SimValueError('For an ODEModel, noise matrix G should be None')
    if G is None or modelType is ODEModel:
        newclass = type(name, (ODEModel,), dict())
        setattr(newclass, 'f', staticmethod(__clone_function(f, 'f')))
    else:
        newclass = type(name, (modelType,), dict())
        setattr(newclass, 'f', staticmethod(__clone_function(f, 'f')))
        setattr(newclass, 'G', staticmethod(__clone_function(G, 'G')))
    setattr(newclass, 'y0', copy.deepcopy(y0))
    # For any global that is used by the functions f or G, create a 
    # corresponding attribute in our new class.
    globals_used = [x for x in f.__globals__ if (x in f.__code__.co_names or 
        G is not None and x in G.__code__.co_names)]
    for x in globals_used:
        if G is None:
            setattr(newclass, x, __AccessDict(x, newclass.f.__globals__))
        else:
            setattr(newclass, x, __AccessDicts(x, newclass.f.__globals__, 
                                                  newclass.G.__globals__))
    # Put the new class into namespace __main__ (to cause dill to pickle it)
    newclass.__module__ = '__main__'
    import __main__
    __main__.__dict__[name] = newclass 
    return newclass