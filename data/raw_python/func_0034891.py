def _replace_match(m, env):
    """Given a match object, having matched something inside curly braces,
    replace the contents if matches one of the supported tox-substitutions."""
    # ditch the curly braces
    s = m.group()[1:-1].strip()

    try:
        # get the env attributes e.g. envpython or toxinidir.
        # Note: if you ask for a env methodname this will raise
        # later on... so don't do that.
        return getattr(env, s)
    except AttributeError:
        pass

    for r in [_replace_envvar, _replace_config, _replace_posargs]:
        try:
            return r(s, env)
        except ValueError:
            pass

    raise NotImplementedError("{%s} not understood in tox.ini file." % s)