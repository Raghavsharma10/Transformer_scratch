def _build_vars_dict(vars_file='', variables=None):
    """Merge variables into a single dictionary

    Applies to CLI provided variables only
    """
    repex_vars = {}
    if vars_file:
        with open(vars_file) as varsfile:
            repex_vars = yaml.safe_load(varsfile.read())
    for var in variables:
        key, value = var.split('=')
        repex_vars.update({str(key): str(value)})
    return repex_vars