def dircolorize(path, name_only=True):
    """Use user dircolors settings to colorize a string which is a path.
    If name_only is True, it does this by the name rules (*.x) only; it
    will not check the filesystem to colorize things like pipes, block devs,
    doors, etc."""
    if not name_only:
        raise NotImplemented("Filesystem checking not implemented.")
    for k,regex in colorremap.iteritems():
        if regex.match(path):
            return '\x1b[%(color)sm%(path)s\x1b[00m' % {'color': k, 'path': path}
    return path