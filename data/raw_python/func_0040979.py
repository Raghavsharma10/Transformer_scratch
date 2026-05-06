def new(ruletype, **kwargs):
    """Instantiate a new build rule based on kwargs.

    Appropriate args list varies with rule type.
    Minimum args required:
      [... fill this in ...]
    """
    try:
        ruleclass = TYPE_MAP[ruletype]
    except KeyError:
        raise error.InvalidRule('Unrecognized rule type: %s' % ruletype)

    try:
        return ruleclass(**kwargs)
    except TypeError:
        log.error('BADNESS. ruletype: %s, data: %s', ruletype, kwargs)
        raise