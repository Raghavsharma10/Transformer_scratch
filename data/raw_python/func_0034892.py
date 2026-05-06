def _replace_envvar(s, _):
    """env:KEY or env:KEY:DEFAULT"""
    e = s.split(":")
    if len(e) > 3 or len(e) == 1 or e[0] != "env":
        raise ValueError()
    elif len(e) == 2:
        # Note: this can/should raise a KeyError (according to spec).
        return os.environ[e[1]]
    else:  # len(e) == 3
        return os.environ.get(e[1], e[2])