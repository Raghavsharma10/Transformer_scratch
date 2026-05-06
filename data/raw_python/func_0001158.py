def query_artifacts(job_ids, log):
    """Query API again for artifacts.

    :param iter job_ids: List of AppVeyor jobIDs.
    :param logging.Logger log: Logger for this function. Populated by with_log() decorator.

    :return: List of tuples: (job ID, artifact file name, artifact file size).
    :rtype: list
    """
    jobs_artifacts = list()
    for job in job_ids:
        url = '/buildjobs/{0}/artifacts'.format(job)
        log.debug('Querying AppVeyor artifact API for %s...', job)
        json_data = query_api(url)
        for artifact in json_data:
            jobs_artifacts.append((job, artifact['fileName'], artifact['size']))
    return jobs_artifacts