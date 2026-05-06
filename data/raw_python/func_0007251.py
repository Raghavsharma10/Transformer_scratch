def format_check(settings):
    """
    Check the format of a osmnet_config object.

    Parameters
    ----------
    settings : dict
        osmnet_config as a dictionary
    Returns
    -------
    Nothing
    """

    valid_keys = ['logs_folder', 'log_file', 'log_console', 'log_name',
                  'log_filename', 'keep_osm_tags']

    for key in list(settings.keys()):
        assert key in valid_keys, \
            ('{} not found in list of valid configuation keys').format(key)
        assert isinstance(key, str), ('{} must be a string').format(key)
        if key == 'keep_osm_tags':
            assert isinstance(settings[key], list), \
                ('{} must be a list').format(key)
            for value in settings[key]:
                assert all(isinstance(element, str) for element in value), \
                    'all elements must be a string'
        if key == 'log_file' or key == 'log_console':
            assert isinstance(settings[key], bool), \
                ('{} must be boolean').format(key)