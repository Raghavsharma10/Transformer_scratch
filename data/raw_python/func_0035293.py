def vsh(cmd, *args, **kw):
    """ Execute a command installed into the active virtualenv.
    """
    args = '" "'.join(i.replace('"', r'\"') for i in args)
    easy.sh('"%s" "%s"' % (venv_bin(cmd), args))