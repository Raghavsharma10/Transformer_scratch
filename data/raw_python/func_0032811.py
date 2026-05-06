def _construct_register(reg, default_reg):
    """Constructs a register dict."""
    if reg:
        x = dict((k, reg.get(k, d)) for k, d in default_reg.items())
    else:
        x = dict(default_reg)
    return x