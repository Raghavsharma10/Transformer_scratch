def save_config(**kwargs):
    """Save configuration keys to vispy config file

    Parameters
    ----------
    **kwargs : keyword arguments
        Key/value pairs to save to the config file.
    """
    if kwargs == {}:
        kwargs = config._config
    current_config = _load_config()
    current_config.update(**kwargs)
    # write to disk
    fname = _get_config_fname()
    if fname is None:
        raise RuntimeError('config filename could not be determined')
    if not op.isdir(op.dirname(fname)):
        os.mkdir(op.dirname(fname))
    with open(fname, 'w') as fid:
        json.dump(current_config, fid, sort_keys=True, indent=0)