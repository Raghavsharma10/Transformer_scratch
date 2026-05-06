def _init():
    """ Create global Config object, parse command flags
    """
    global config, _data_path, _allowed_config_keys

    app_dir = _get_vispy_app_dir()
    if app_dir is not None:
        _data_path = op.join(app_dir, 'data')
        _test_data_path = op.join(app_dir, 'test_data')
    else:
        _data_path = _test_data_path = None

    # All allowed config keys and the types they may have
    _allowed_config_keys = {
        'data_path': string_types,
        'default_backend': string_types,
        'gl_backend': string_types,
        'gl_debug': (bool,),
        'glir_file': string_types+file_types,
        'include_path': list,
        'logging_level': string_types,
        'qt_lib': string_types,
        'dpi': (int, type(None)),
        'profile': string_types + (type(None),),
        'audit_tests': (bool,),
        'test_data_path': string_types + (type(None),),
    }

    # Default values for all config options
    default_config_options = {
        'data_path': _data_path,
        'default_backend': '',
        'gl_backend': 'gl2',
        'gl_debug': False,
        'glir_file': '',
        'include_path': [],
        'logging_level': 'info',
        'qt_lib': 'any',
        'dpi': None,
        'profile': None,
        'audit_tests': False,
        'test_data_path': _test_data_path,
    }

    config = Config(**default_config_options)

    try:
        config.update(**_load_config())
    except Exception as err:
        raise Exception('Error while reading vispy config file "%s":\n  %s' %
                        (_get_config_fname(), err.message))
    set_log_level(config['logging_level'])

    _parse_command_line_arguments()