def rime_solver_cfg(**kwargs):
    """
    Produces a SolverConfiguration object, inherited from
    a simple python dict, and containing the options required
    to configure the RIME Solver.

    Keyword arguments
    -----------------
    Any keyword arguments are inserted into the
    returned dict.

    Returns
    -------
    A SolverConfiguration object.
    """
    from configuration import (load_config, config_validator,
        raise_validator_errors)

    def _merge_copy(d1, d2):
        return { k: _merge_copy(d1[k], d2[k]) if k in d1
                                                and isinstance(d1[k], dict)
                                                and isinstance(d2[k], dict)
                                            else d2[k] for k in d2 }

    try:
        cfg_file = kwargs.pop('cfg_file')
    except KeyError as e:
        slvr_cfg = kwargs
    else:
        cfg = load_config(cfg_file)
        slvr_cfg = _merge_copy(cfg, kwargs)

    # Validate the configuration, raising any errors
    validator = config_validator()
    validator.validate(slvr_cfg)
    raise_validator_errors(validator)

    return validator.document