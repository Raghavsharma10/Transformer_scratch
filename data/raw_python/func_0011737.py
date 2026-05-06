def newsim(f, G, y0, name='NewModel', modelType=ItoModel, T=60.0, dt=0.005, repeat=1, identical=True):
    """Make a simulation of the system defined by functions f and G.

    dy = f(y,t)dt + G(y,t).dW with initial condition y0
    This helper function is for convenience, making it easy to define 
    one-off simulations interactively in ipython.

    Args:
      f: callable(y, t) (defined in global scope) returning (n,) array
        Vector-valued function to define the deterministic part of the system 
      G: callable(y, t) (defined in global scope) returning (n,m) array
        Optional matrix-valued function to define noise coefficients of an Ito
        SDE system.
      y0 (array):  Initial condition 
      name (str): Optional class name for the new model
      modelType (type): The type of model to simulate. Must be a subclass of
        nsim.Model, for example nsim.ODEModel, nsim.ItoModel or 
        nsim.StratonovichModel. The default is nsim.ItoModel.
      T: Total length of time to simulate, in seconds.
      dt: Timestep for numerical integration.
      repeat (int, optional)
      identical (bool, optional)

    Returns: 
      Simulation

    Raises:
      SimValueError, SimTypeError
    """
    NewModel = newmodel(f, G, y0, name, modelType)
    if repeat == 1:
        return Simulation(NewModel(), T, dt)
    else:
        return RepeatedSim(NewModel, T, dt, repeat, identical)