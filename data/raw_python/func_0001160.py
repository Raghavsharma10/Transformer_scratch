def get_urls(config, log):
    """Wait for AppVeyor job to finish and get all artifacts' URLs.

    :param dict config: Dictionary from get_arguments().
    :param logging.Logger log: Logger for this function. Populated by with_log() decorator.

    :return: Paths and URLs from artifacts_urls.
    :rtype: dict
    """
    # Wait for job to be queued. Once it is we'll have the "version".
    build_version = None
    for _ in range(3):
        build_version = query_build_version(config)
        if build_version:
            break
        log.info('Waiting for job to be queued...')
        time.sleep(SLEEP_FOR)
    if not build_version:
        log.error('Timed out waiting for job to be queued or build not found.')
        raise HandledError

    # Get job IDs. Wait for AppVeyor job to finish.
    job_ids = list()
    valid_statuses = ['success', 'failed', 'running', 'queued']
    while True:
        job_ids = query_job_ids(build_version, config)
        statuses = set([i[1] for i in job_ids])
        if 'failed' in statuses:
            job = [i[0] for i in job_ids if i[1] == 'failed'][0]
            url = 'https://ci.appveyor.com/project/{0}/{1}/build/job/{2}'.format(config['owner'], config['repo'], job)
            log.error('AppVeyor job failed: %s', url)
            raise HandledError
        if statuses == set(valid_statuses[:1]):
            log.info('Build successful. Found %d job%s.', len(job_ids), '' if len(job_ids) == 1 else 's')
            break
        if 'running' in statuses:
            log.info('Waiting for job%s to finish...', '' if len(job_ids) == 1 else 's')
        elif 'queued' in statuses:
            log.info('Waiting for all jobs to start...')
        else:
            log.error('Got unknown status from AppVeyor API: %s', ' '.join(statuses - set(valid_statuses)))
            raise HandledError
        time.sleep(SLEEP_FOR)

    # Get artifacts.
    artifacts = query_artifacts([i[0] for i in job_ids])
    log.info('Found %d artifact%s.', len(artifacts), '' if len(artifacts) == 1 else 's')
    return artifacts_urls(config, artifacts) if artifacts else dict()