def prepare_model(module: Union[types.ModuleType, str],
                  timestep: PeriodABC.ConstrArg = None):
    """Prepare and return the model of the given module.

    In usual HydPy projects, each hydrological model instance is prepared
    in an individual control file.  This allows for "polluting" the
    namespace with different model attributes.  There is no danger of
    name conflicts, as long as no other (wildcard) imports are performed.

    However, there are situations when different models are to be loaded
    into the same namespace.  Then it is advisable to use function
    |prepare_model|, which just returns a reference to the model
    and nothing else.

    See the documentation of |dam_v001| on how to apply function
    |prepare_model| properly.
    """
    if timestep is not None:
        parametertools.Parameter.parameterstep(timetools.Period(timestep))
    try:
        model = module.Model()
    except AttributeError:
        module = importlib.import_module(f'hydpy.models.{module}')
        model = module.Model()
    if hydpy.pub.options.usecython and hasattr(module, 'cythonizer'):
        cymodule = module.cythonizer.cymodule
        cymodel = cymodule.Model()
        cymodel.parameters = cymodule.Parameters()
        cymodel.sequences = cymodule.Sequences()
        model.cymodel = cymodel
        for numpars_name in ('NumConsts', 'NumVars'):
            if hasattr(cymodule, numpars_name):
                numpars_new = getattr(cymodule, numpars_name)()
                numpars_old = getattr(model, numpars_name.lower())
                for (name_numpar, numpar) in vars(numpars_old).items():
                    setattr(numpars_new, name_numpar, numpar)
                setattr(cymodel, numpars_name.lower(), numpars_new)
        for name in dir(cymodel):
            if (not name.startswith('_')) and hasattr(model, name):
                setattr(model, name, getattr(cymodel, name))
        dict_ = {'cythonmodule': cymodule,
                 'cymodel': cymodel}
    else:
        dict_ = {}
    dict_.update(vars(module))
    dict_['model'] = model
    if hasattr(module, 'Parameters'):
        model.parameters = module.Parameters(dict_)
    else:
        model.parameters = parametertools.Parameters(dict_)
    if hasattr(module, 'Sequences'):
        model.sequences = module.Sequences(**dict_)
    else:
        model.sequences = sequencetools.Sequences(**dict_)
    if hasattr(module, 'Masks'):
        model.masks = module.Masks(model)
    return model