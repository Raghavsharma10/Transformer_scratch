def _get_config_fname():
    """Helper for the vispy config file"""
    directory = _get_vispy_app_dir()
    if directory is None:
        return None
    fname = op.join(directory, 'vispy.json')
    if os.environ.get('_VISPY_CONFIG_TESTING', None) is not None:
        fname = op.join(_TempDir(), 'vispy.json')
    return fname