def cpjoin(*args):
    """ custom path join """
    rooted = True if args[0].startswith('/') else False
    def deslash(a): return a[1:] if a.startswith('/') else a
    newargs = [deslash(arg) for arg in args]
    path = os.path.join(*newargs)
    if rooted: path = os.path.sep + path
    return path