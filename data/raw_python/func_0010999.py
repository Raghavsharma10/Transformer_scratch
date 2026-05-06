def checks():
    """
    An iterator of valid checks that are in the installed checkers package.

    yields check name, check module
    """
    checkers_dir = os.path.dirname(checkers.__file__)
    mod_names = [name for _, name, _ in pkgutil.iter_modules([checkers_dir])]
    for name in mod_names:
        mod = importlib.import_module("checkers.{0}".format(name))

        # Does the module have a "run" function
        if isinstance(getattr(mod, 'run', None), types.FunctionType):
            # has a run method, yield it
            yield getattr(mod, 'CHECK_NAME', name), mod