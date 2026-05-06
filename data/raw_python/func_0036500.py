def cancel_job(agent, project_name, job_id):
    """
    cancel a job.
    If the job is pending, it will be removed. If the job is running, it will be terminated.
    """
    prevstate = agent.cancel(project_name, job_id)['prevstate']
    if prevstate == 'pending':
        sqllite_agent.execute(ScrapydJobExtInfoSQLSet.DELETE_BY_ID, (job_id,))