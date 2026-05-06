def hatchery():
    """ Main entry point for the hatchery program """
    args = docopt.docopt(__doc__)
    task_list = args['<task>']

    if not task_list or 'help' in task_list or args['--help']:
        print(__doc__.format(version=_version.__version__, config_files=config.CONFIG_LOCATIONS))
        return 0

    level_str = args['--log-level']
    try:
        level_const = getattr(logging, level_str.upper())
        logging.basicConfig(level=level_const)
        if level_const == logging.DEBUG:
            workdir.options.debug = True
    except LookupError:
        logging.basicConfig()
        logger.error('received invalid log level: ' + level_str)
        return 1

    for task in task_list:
        if task not in ORDERED_TASKS:
            logger.info('starting task: check')
            logger.error('received invalid task: ' + task)
            return 1

    for task in CHECK_TASKS:
        if task in task_list:
            task_check(args)
            break

    if 'package' in task_list and not args['--release-version']:
        logger.error('--release-version is required for the package task')
        return 1

    config_dict = _get_config_or_die(
        calling_task='hatchery',
        required_params=['auto_push_tag']
    )
    if config_dict['auto_push_tag'] and 'upload' in task_list:
        logger.info('adding task: tag (auto_push_tag==True)')
        task_list.append('tag')

    # all commands will raise a SystemExit if they fail
    # check will have already been run
    for task in ORDERED_TASKS:
        if task in task_list and task != 'check':
            logger.info('starting task: ' + task)
            globals()['task_' + task](args)

    logger.info("all's well that ends well...hatchery out")
    return 0