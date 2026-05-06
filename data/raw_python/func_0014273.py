def prepare_system_options(cfg, defaults=None):
    """
    Retrieve and delete (pop) system options from input configuration.
    """
    d = {} if defaults is None else defaults.copy()
    if 'nonbondedMethod' in cfg:
        d['nonbondedMethod'] = warned_getattr(openmm_app, cfg.pop('nonbondedMethod'), None)
    if 'nonbondedCutoff' in cfg:
        d['nonbondedCutoff'] = cfg.pop('nonbondedCutoff') * u.nanometers
    if 'constraints' in cfg:
        d['constraints'] = warned_getattr(openmm_app, cfg.pop('constraints'), None)
    for key in ['rigidWater', 'ewaldErrorTolerance']:
        if key in cfg:
            d[key] = cfg.pop(key)
    if 'extra_system_options' in cfg:
        if 'implicitSolvent' in cfg['extra_system_options']:
            implicit_solvent = warned_getattr(
                openmm_app, cfg['extra_system_options']['implicitSolvent'], None)
            cfg['extra_system_options']['implicitSolvent'] = implicit_solvent
        d.update(cfg.pop('extra_system_options'))
    return d