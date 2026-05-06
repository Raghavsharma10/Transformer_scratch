def get_version():
    """Obtain the version number"""
    import imp
    import os
    mod = imp.load_source(
        'version', os.path.join('skdata', '__init__.py')
    )
    return mod.__version__