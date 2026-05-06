def Versions():
    """Returns a string with version information.

    You would call this function if you want a string giving detailed information
    on the version of ``phydms`` and the associated packages that it uses.
    """
    s = [\
            'Version information:',
            '\tTime and date: %s' % time.asctime(),
            '\tPlatform: %s' % platform.platform(),
            '\tPython version: %s' % sys.version.replace('\n', ' '),
            '\tphydms version: %s' % phydmslib.__version__,
            ]
    for modname in ['Bio', 'cython', 'numpy', 'scipy', 'matplotlib',
            'natsort', 'sympy', 'six', 'pandas', 'pyvolve', 'statsmodels',
            'weblogolib', 'PyPDF2']:
        try:
            v = importlib.import_module(modname).__version__
            s.append('\t%s version: %s' % (modname, v))
        except ImportError:
            s.append('\t%s cannot be imported into Python' % modname)
    return '\n'.join(s)