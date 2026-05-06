def _load_config():
    """Helper to load prefs from ~/.vispy/vispy.json"""
    fname = _get_config_fname()
    if fname is None or not op.isfile(fname):
        return dict()
    with open(fname, 'r') as fid:
        config = json.load(fid)
    return config