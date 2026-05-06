def parse_args(args, kwargs):
    """Returns a kwargs dictionary by turning args into kwargs"""
    if 'style' in kwargs:
        args += (kwargs['style'],)
        del kwargs['style']
    for arg in args:
        if not isinstance(arg, (bytes, unicode)):
            raise ValueError("args must be strings:" + repr(args))
        if arg.lower() in FG_COLORS:
            if 'fg' in kwargs: raise ValueError("fg specified twice")
            kwargs['fg'] = FG_COLORS[arg]
        elif arg.lower().startswith('on_') and arg[3:].lower() in BG_COLORS:
            if 'bg' in kwargs: raise ValueError("fg specified twice")
            kwargs['bg'] = BG_COLORS[arg[3:]]
        elif arg.lower() in STYLES:
            kwargs[arg] = True
        else:
            raise ValueError("couldn't process arg: "+repr(arg))
    for k in kwargs:
        if k not in ['fg', 'bg'] + list(STYLES.keys()):
            raise ValueError("Can't apply that transformation")
    if 'fg' in kwargs:
        if kwargs['fg'] in FG_COLORS:
            kwargs['fg'] = FG_COLORS[kwargs['fg']]
        if kwargs['fg'] not in list(FG_COLORS.values()):
            raise ValueError("Bad fg value: %r" % kwargs['fg'])
    if 'bg' in kwargs:
        if kwargs['bg'] in BG_COLORS:
            kwargs['bg'] = BG_COLORS[kwargs['bg']]
        if kwargs['bg'] not in list(BG_COLORS.values()):
            raise ValueError("Bad bg value: %r" % kwargs['bg'])
    return kwargs