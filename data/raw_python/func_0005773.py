def handle_path(pathobj, variables=None, diff=False):
    """Iterate over all chosen files in a path

    :param dict pathobj: a dict of a specific path in the config
    :param dict variables: a dict of variables (can be None)
    """
    logger.info('Handling path with description: %s',
                pathobj.get('description'))

    variables = variables or {}
    variable_expander = _VariablesHandler()
    pathobj = variable_expander.expand(variables, pathobj)

    pathobj = _set_path_defaults(pathobj)

    path_to_handle = os.path.join(pathobj['base_directory'], pathobj['path'])
    logger.debug('Path to process: %s', path_to_handle)

    validate = 'validator' in pathobj
    if validate:
        validator_config = pathobj['validator']
        validator = _Validator(validator_config)
        validator_type = validator_config.get('type', 'per_type')

    rpx = Repex(pathobj)

    if not pathobj.get('type'):
        _handle_single_file(
            rpx=rpx,
            path_to_handle=path_to_handle,
            pathobj=pathobj,
            validate=validate,
            diff=diff,
            validator=validator if validate else None)
    else:
        _handle_multiple_files(
            rpx=rpx,
            path_to_handle=path_to_handle,
            pathobj=pathobj,
            validate=validate,
            diff=diff,
            validator=validator if validate else None,
            validator_type=validator_type if validate else None)