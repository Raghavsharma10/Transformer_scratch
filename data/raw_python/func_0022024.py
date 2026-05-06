def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    if domain != 'py':
        return None
    modname = info['module']
    fullname = info['fullname']
    submod = sys.modules.get(modname)
    if submod is None:
        return None
    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except:
            return None
    try:
        fn = inspect.getsourcefile(obj)
    except:
        fn = None
    if not fn:
        return None
    try:
        source, lineno = inspect.findsource(obj)
    except:
        lineno = None
    if lineno:
        linespec = "#L%d" % (lineno + 1)
    else:
        linespec = ""
    fn = relpath(fn, start='..')
    return "https://github.com/mithrandi/txacme/blob/%s/%s%s" % (
        txacme_version_info['full-revisionid'], fn, linespec)