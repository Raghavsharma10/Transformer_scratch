def validate(config, log):
    """Validate config values.

    :raise HandledError: On invalid config values.

    :param dict config: Dictionary from get_arguments().
    :param logging.Logger log: Logger for this function. Populated by with_log() decorator.
    """
    if config['always_job_dirs'] and config['no_job_dirs']:
        log.error('Contradiction: --always-job-dirs and --no-job-dirs used.')
        raise HandledError
    if config['commit'] and not REGEX_COMMIT.match(config['commit']):
        log.error('No or invalid git commit obtained.')
        raise HandledError
    if config['dir'] and not os.path.isdir(config['dir']):
        log.error("Not a directory or doesn't exist: %s", config['dir'])
        raise HandledError
    if config['no_job_dirs'] not in ('', 'rename', 'overwrite', 'skip'):
        log.error('--no-job-dirs has invalid value. Check --help for valid values.')
        raise HandledError
    if not config['owner'] or not REGEX_GENERAL.match(config['owner']):
        log.error('No or invalid repo owner name obtained.')
        raise HandledError
    if config['pull_request'] and not config['pull_request'].isdigit():
        log.error('--pull-request is not a digit.')
        raise HandledError
    if not config['repo'] or not REGEX_GENERAL.match(config['repo']):
        log.error('No or invalid repo name obtained.')
        raise HandledError
    if config['tag'] and not REGEX_GENERAL.match(config['tag']):
        log.error('Invalid git tag obtained.')
        raise HandledError