def stop_workflow(config, *, names=None):
    """ Stop one or more workflows.

    Args:
        config (Config): Reference to the configuration object from which the
            settings for the workflow are retrieved.
        names (list): List of workflow names, workflow ids or workflow job ids for the
            workflows that should be stopped. If all workflows should be
            stopped, set it to None.

    Returns:
        tuple: A tuple of the workflow jobs that were successfully stopped and the ones
            that could not be stopped.
    """
    jobs = list_jobs(config, filter_by_type=JobType.Workflow)

    if names is not None:
        filtered_jobs = []
        for job in jobs:
            if (job.id in names) or (job.name in names) or (job.workflow_id in names):
                filtered_jobs.append(job)
    else:
        filtered_jobs = jobs

    success = []
    failed = []
    for job in filtered_jobs:
        client = Client(SignalConnection(**config.signal, auto_connect=True),
                        request_key=job.workflow_id)

        if client.send(Request(action='stop_workflow')).success:
            success.append(job)
        else:
            failed.append(job)

    return success, failed