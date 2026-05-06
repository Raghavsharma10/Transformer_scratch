def load_py_config(conf_file):
    # type: (str) -> None
    """ Import configuration from a python file.

    This will just import the file into python. Sky is the limit. The file
    has to deal with the configuration all by itself (i.e. call conf.init()).
    You will also need to add your src directory to sys.paths if it's not the
    current working directory. This is done automatically if you use yaml
    config as well.

    Args:
        conf_file (str):
            Path to the py module config. This function will not check the file
            name or extension and will just crash if the given file does not
            exist or is not a valid python file.
    """
    if sys.version_info >= (3, 5):
        from importlib import util

        spec = util.spec_from_file_location('pelconf', conf_file)
        mod = util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    elif sys.version_info >= (3, 3):
        from importlib import machinery
        loader = machinery.SourceFileLoader('pelconf', conf_file)
        _ = loader.load_module()

    elif sys.version_info <= (3, 0):
        import imp

        imp.load_source('pelconf', conf_file)