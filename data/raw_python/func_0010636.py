def list_jobs(config, *, status=JobStatus.Active,
              filter_by_type=None, filter_by_worker=None):
    """ Return a list of Celery jobs.

    Args:
        config (Config): Reference to the configuration object from which the
            settings are retrieved.
        status (JobStatus): The status of the jobs that should be returned.
        filter_by_type (list): Restrict the returned jobs to the types in this list.
        filter_by_worker (list): Only return jobs that were registered, reserved or are
            running on the workers given in this list of worker names. Using
            this option will increase the performance.

    Returns:
        list: A list of JobStats.
    """
    celery_app = create_app(config)

    # option to filter by the worker (improves performance)
    if filter_by_worker is not None:
        inspect = celery_app.control.inspect(
            destination=filter_by_worker if isinstance(filter_by_worker, list)
            else [filter_by_worker])
    else:
        inspect = celery_app.control.inspect()

    # get active, registered or reserved jobs
    if status == JobStatus.Active:
        job_map = inspect.active()
    elif status == JobStatus.Registered:
        job_map = inspect.registered()
    elif status == JobStatus.Reserved:
        job_map = inspect.reserved()
    elif status == JobStatus.Scheduled:
        job_map = inspect.scheduled()
    else:
        job_map = None

    if job_map is None:
        return []

    result = []
    for worker_name, jobs in job_map.items():
        for job in jobs:
            try:
                job_stats = JobStats.from_celery(worker_name, job, celery_app)

                if (filter_by_type is None) or (job_stats.type == filter_by_type):
                    result.append(job_stats)
            except JobStatInvalid:
                pass

    return result