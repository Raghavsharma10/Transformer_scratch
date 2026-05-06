def get_all_job_list(agent):
    """
    Get all job list by each project name then
    return three job list on the base of different status(pending,running,finished).
    """
    project_list = agent.get_project_list()
    if project_list['status'] == 'error':
        raise ScrapydTimeoutException
    project_list = project_list['projects']
    pending_job_list = []
    running_job_list = []
    finished_job_list = []
    for project_name in project_list:
        job_list = agent.get_job_list(project_name)
        # Extract latest version
        project_version = agent.get_version_list(project_name)['versions'][-1:]
        for pending_job in job_list['pending']:
            pending_job_list.append(JobListDO(project_name=project_name,
                                              project_version=project_version,
                                              job_id=pending_job['id'],
                                              spider_name=pending_job['spider'],
                                              job_status=JobStatus.PENDING
                                              ))
        for running_job in job_list['running']:
            running_job_list.append(JobListDO(project_name=project_name,
                                              project_version=project_version,
                                              job_id=running_job['id'],
                                              spider_name=running_job['spider'],
                                              start_time=running_job['start_time'],
                                              job_status=JobStatus.RUNNING
                                              ))
        for finished_job in job_list['finished']:
            finished_job_list.append(JobListDO(project_name=project_name,
                                               project_version=project_version,
                                               job_id=finished_job['id'],
                                               spider_name=finished_job['spider'],
                                               start_time=finished_job['start_time'],
                                               end_time=finished_job['end_time'],
                                               job_status=JobStatus.FINISHED
                                               ))

    return pending_job_list, running_job_list, finished_job_list