def runProcess(cmd, *args):
    """Run `cmd` (which is searched for in the executable path) with `args` and
    return the exit status.

    In general (unless you know what you're doing) use::

     runProcess('program', filename)

    rather than::

     os.system('program %s' % filename)

    because the latter will not work as expected if `filename` contains
    spaces or shell-metacharacters.

    If you need more fine-grained control look at ``os.spawn*``.
    """
    from os import spawnvp, P_WAIT
    return spawnvp(P_WAIT, cmd, (cmd,) + args)