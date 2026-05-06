def align_options(options):
    """
    Indents flags and aligns help texts.
    """
    l = 0
    for opt in options:
        if len(opt[0]) > l:
            l = len(opt[0])
    s = []
    for opt in options:
        s.append('  {0}{1}  {2}'.format(opt[0], ' ' * (l - len(opt[0])), opt[1]))
    return '\n'.join(s)