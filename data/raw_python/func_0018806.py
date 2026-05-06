def autodoc_basemodel(module):
    """Add an exhaustive docstring to the given module of a basemodel.

    Works onlye when all modules of the basemodel are named in the
    standard way, e.g. `lland_model`, `lland_control`, `lland_inputs`.
    """
    autodoc_tuple2doc(module)
    namespace = module.__dict__
    doc = namespace.get('__doc__')
    if doc is None:
        doc = ''
    basemodulename = namespace['__name__'].split('.')[-1]
    modules = {key: value for key, value in namespace.items()
               if (isinstance(value, types.ModuleType) and
                   key.startswith(basemodulename+'_'))}
    substituter = Substituter(hydpy.substituter)
    lines = []
    specification = 'model'
    modulename = basemodulename+'_'+specification
    if modulename in modules:
        module = modules[modulename]
        lines += _add_title('Model features', '-')
        lines += _add_lines(specification, module)
        substituter.add_module(module)
    for (title, spec2capt) in (('Parameter features', _PAR_SPEC2CAPT),
                               ('Sequence features', _SEQ_SPEC2CAPT),
                               ('Auxiliary features', _AUX_SPEC2CAPT)):
        found_module = False
        new_lines = _add_title(title, '-')
        for (specification, caption) in spec2capt.items():
            modulename = basemodulename+'_'+specification
            module = modules.get(modulename)
            if module:
                found_module = True
                new_lines += _add_title(caption, '.')
                new_lines += _add_lines(specification, module)
                substituter.add_module(module)
        if found_module:
            lines += new_lines
    doc += '\n'.join(lines)
    namespace['__doc__'] = doc
    basemodule = importlib.import_module(namespace['__name__'])
    substituter.add_module(basemodule)
    substituter.update_masters()
    namespace['substituter'] = substituter