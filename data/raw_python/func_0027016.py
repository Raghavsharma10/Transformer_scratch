def get_kwargs_defaults(argspec):
    """Computes a kwargs_defaults dictionary for use by get_args_tuple given an argspec."""
    arg_names = tuple(argspec.args)
    defaults = argspec.defaults or ()
    num_args = len(argspec.args) - len(defaults)
    kwargs_defaults = {}
    for i, default_value in enumerate(defaults):
        kwargs_defaults[arg_names[num_args + i]] = default_value
    if getattr(argspec, "kwonlydefaults", None):
        kwargs_defaults.update(argspec.kwonlydefaults)
    return kwargs_defaults