def iterate(config_file_path=None,
            config=None,
            variables=None,
            tags=None,
            validate=True,
            validate_only=False,
            with_diff=False):
    """Iterate over all paths in `config_file_path`

    :param string config_file_path: a path to a repex config file
    :param dict config: a dictionary representing a repex config
    :param dict variables: a dict of variables (can be None)
    :param list tags: a list of tags to check for
    :param bool validate: whether to perform schema validation on the config
    :param bool validate_only: only perform validation without running
    :param bool with_diff: whether to write a diff of all changes to a file
    """
    # TODO: Check if tags can be a tuple instead of a list
    if not isinstance(variables or {}, dict):
        raise TypeError(ERRORS['variables_not_dict'])
    if not isinstance(tags or [], list):
        raise TypeError(ERRORS['tags_not_list'])

    config = _get_config(config_file_path, config)
    if validate or validate_only:
        _validate_config_schema(config)
    if validate_only:
        logger.info('Config file validation completed successfully!')
        sys.exit(0)

    repex_vars = _merge_variables(config['variables'], variables or {})
    repex_tags = tags or []
    logger.debug('Chosen tags: %s', repex_tags)

    for path in config['paths']:
        _process_path(path, repex_tags, repex_vars, with_diff)