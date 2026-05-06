def query_job_ids(build_version, config, log):
    """Get one or more job IDs and their status associated with a build version.

    Filters jobs by name if --job-name is specified.

    :raise HandledError: On invalid JSON data or bad job name.

    :param str build_version: AppVeyor build version from query_build_version().
    :param dict config: Dictionary from get_arguments().
    :param logging.Logger log: Logger for this function. Populated by with_log() decorator.

    :return: List of two-item tuples. Job ID (first) and its status (second).
    :rtype: list
    """
    url = '/projects/{0}/{1}/build/{2}'.format(config['owner'], config['repo'], build_version)

    # Query version.
    log.debug('Querying AppVeyor version API for %s/%s at %s...', config['owner'], config['repo'], build_version)
    json_data = query_api(url)
    if 'build' not in json_data:
        log.error('Bad JSON reply: "build" key missing.')
        raise HandledError
    if 'jobs' not in json_data['build']:
        log.error('Bad JSON reply: "jobs" key missing.')
        raise HandledError

    # Find AppVeyor job.
    all_jobs = list()
    for job in json_data['build']['jobs']:
        if config['job_name'] and config['job_name'] == job['name']:
            log.debug('Filtering by job name: found match!')
            return [(job['jobId'], job['status'])]
        all_jobs.append((job['jobId'], job['status']))
    if config['job_name']:
        log.error('Job name "%s" not found.', config['job_name'])
        raise HandledError
    return all_jobs