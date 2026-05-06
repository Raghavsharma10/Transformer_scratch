def load_yaml_config(conf_file):
    # type: (str) -> None
    """ Load a YAML configuration.

    This will not update the configuration but replace it entirely.

    Args:
        conf_file (str):
            Path to the YAML config. This function will not check the file name
            or extension and will just crash if the given file does not exist or
            is not a valid YAML file.
    """
    global g_config

    with open(conf_file) as fp:
        # Initialize config
        g_config = util.yaml_load(fp)

        # Add src_dir to sys.paths if it's set. This is only done with YAML
        # configs, py configs have to do this manually.
        src_dir = get_path('src_dir', None)
        if src_dir is not None:
            sys.path.insert(0, src_dir)

        for cmd in get('commands', []):
            _import(cmd)