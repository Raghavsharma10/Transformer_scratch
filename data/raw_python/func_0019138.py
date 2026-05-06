def parameterstep(timestep=None):
    """Define a parameter time step size within a parameter control file.

    Argument:
      * timestep(|Period|): Time step size.

    Function parameterstep should usually be be applied in a line
    immediately behind the model import.  Defining the step size of time
    dependent parameters is a prerequisite to access any model specific
    parameter.

    Note that parameterstep implements some namespace magic by
    means of the module |inspect|.  This makes things a little
    complicated for framework developers, but it eases the definition of
    parameter control files for framework users.
    """
    if timestep is not None:
        parametertools.Parameter.parameterstep(timestep)
    namespace = inspect.currentframe().f_back.f_locals
    model = namespace.get('model')
    if model is None:
        model = namespace['Model']()
        namespace['model'] = model
        if hydpy.pub.options.usecython and 'cythonizer' in namespace:
            cythonizer = namespace['cythonizer']
            namespace['cythonmodule'] = cythonizer.cymodule
            model.cymodel = cythonizer.cymodule.Model()
            namespace['cymodel'] = model.cymodel
            model.cymodel.parameters = cythonizer.cymodule.Parameters()
            model.cymodel.sequences = cythonizer.cymodule.Sequences()
            for numpars_name in ('NumConsts', 'NumVars'):
                if hasattr(cythonizer.cymodule, numpars_name):
                    numpars_new = getattr(cythonizer.cymodule, numpars_name)()
                    numpars_old = getattr(model, numpars_name.lower())
                    for (name_numpar, numpar) in vars(numpars_old).items():
                        setattr(numpars_new, name_numpar, numpar)
                    setattr(model.cymodel, numpars_name.lower(), numpars_new)
            for name in dir(model.cymodel):
                if (not name.startswith('_')) and hasattr(model, name):
                    setattr(model, name, getattr(model.cymodel, name))
        if 'Parameters' not in namespace:
            namespace['Parameters'] = parametertools.Parameters
        model.parameters = namespace['Parameters'](namespace)
        if 'Sequences' not in namespace:
            namespace['Sequences'] = sequencetools.Sequences
        model.sequences = namespace['Sequences'](**namespace)
        namespace['parameters'] = model.parameters
        for pars in model.parameters:
            namespace[pars.name] = pars
        namespace['sequences'] = model.sequences
        for seqs in model.sequences:
            namespace[seqs.name] = seqs
        if 'Masks' in namespace:
            model.masks = namespace['Masks'](model)
            namespace['masks'] = model.masks
    try:
        namespace.update(namespace['CONSTANTS'])
    except KeyError:
        pass
    focus = namespace.get('focus')
    for par in model.parameters.control:
        try:
            if (focus is None) or (par is focus):
                namespace[par.name] = par
            else:
                namespace[par.name] = lambda *args, **kwargs: None
        except AttributeError:
            pass