def load():
    # type: () -> None
    """ Load configuration from file.

    This will search the directory structure upwards to find the project root
    (directory containing ``pelconf.py`` file). Once found it will import the
    config file which should initialize all the configuration (using
    `peltak.core.conf.init()` function).

    You can also have both yaml (configuration) and python (custom commands)
    living together. Just remember that calling `conf.init()` will overwrite
    the config defined in YAML.
    """
    with within_proj_dir():
        if os.path.exists('pelconf.yaml'):
            load_yaml_config('pelconf.yaml')

        if os.path.exists('pelconf.py'):
            load_py_config('pelconf.py')