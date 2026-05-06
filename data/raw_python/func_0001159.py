def artifacts_urls(config, jobs_artifacts, log):
    """Determine destination file paths for job artifacts.

    :param dict config: Dictionary from get_arguments().
    :param iter jobs_artifacts: List of job artifacts from query_artifacts().
    :param logging.Logger log: Logger for this function. Populated by with_log() decorator.

    :return: Destination file paths (keys), download URLs (value[0]), and expected file size (value[1]).
    :rtype: dict
    """
    artifacts = dict()

    # Determine if we should create job ID directories.
    if config['always_job_dirs']:
        job_dirs = True
    elif config['no_job_dirs']:
        job_dirs = False
    elif len(set(i[0] for i in jobs_artifacts)) == 1:
        log.debug('Only one job ID, automatically setting job_dirs = False.')
        job_dirs = False
    elif len(set(i[1] for i in jobs_artifacts)) == len(jobs_artifacts):
        log.debug('No local file conflicts, automatically setting job_dirs = False')
        job_dirs = False
    else:
        log.debug('Multiple job IDs with file conflicts, automatically setting job_dirs = True')
        job_dirs = True

    # Get final URLs and destination file paths.
    root_dir = config['dir'] or os.getcwd()
    for job, file_name, size in jobs_artifacts:
        artifact_url = '{0}/buildjobs/{1}/artifacts/{2}'.format(API_PREFIX, job, file_name)
        artifact_local = os.path.join(root_dir, job if job_dirs else '', file_name)
        if artifact_local in artifacts:
            if config['no_job_dirs'] == 'skip':
                log.debug('Skipping %s from %s', artifact_local, artifact_url)
                continue
            if config['no_job_dirs'] == 'rename':
                new_name = artifact_local
                while new_name in artifacts:
                    path, ext = os.path.splitext(new_name)
                    new_name = (path + '_' + ext) if ext else (new_name + '_')
                log.debug('Renaming %s to %s from %s', artifact_local, new_name, artifact_url)
                artifact_local = new_name
            elif config['no_job_dirs'] == 'overwrite':
                log.debug('Overwriting %s from %s with %s', artifact_local, artifacts[artifact_local][0], artifact_url)
            else:
                log.error('Collision: %s from %s and %s', artifact_local, artifacts[artifact_local][0], artifact_url)
                raise HandledError
        artifacts[artifact_local] = (artifact_url, size)

    return artifacts