def init(app, register_blueprint=True, url_prefix='/fm', access_control_function=None,
         custom_config_json_path=None, custom_init_js_path=None):
    """
    :param app: The Flask app
    :param register_blueprint: Override to False to stop the blueprint from automatically being registered to the
                               app
    :param url_prefix: The URL prefix for the blueprint, defaults to /fm
    :param access_control_function: Pass in a function here to implement access control.  The function will
                                    be called any time someone tries to access the filemanager, and a 404
                                    will be returned if this function returns False
    :param custom_config_json_path: Set this to the full path of you filemanager.config.json file if you want
                                    to use a custom config. Example:
                                    os.path.join(app.root_path, 'static/filemanager.config.json')
    :param custom_init_js_path: Set this to the full path of you filemanager.init.js file if you want
                                to use a custom init.js. Example:
                                os.path.join(app.root_path, 'static/filemanager.init.js')
    """

    global _initialised, _FILE_PATH, _URL_PREFIX

    if _initialised:
        raise Exception('Flask Filemanager can only be registered once!')

    _initialised = True

    _FILE_PATH = app.config.get('FLASKFILEMANAGER_FILE_PATH')
    if not _FILE_PATH:
        raise Exception('No FLASKFILEMANAGER_FILE_PATH value in config')
        
    log.info('File Manager Using file path: {}'.format(_FILE_PATH))
    util.ensure_dir(_FILE_PATH)
    
    if access_control_function:
        set_access_control_function(access_control_function)

    if custom_config_json_path:
        set_custom_config_json_path(custom_config_json_path)
        log.info('File Manager using custom config.json path: {}'.format(custom_config_json_path))

    if custom_init_js_path:
        set_custom_init_js_path(custom_init_js_path)
        log.info('File Manager using custom init.js path: {}'.format(custom_init_js_path))

    if register_blueprint:
        log.info('Registering filemanager blueprint to {}'.format(url_prefix))
        app.register_blueprint(filemanager_blueprint, url_prefix=url_prefix)