def queue(self, new_job_id = None, new_job_name = None, queue_name = None):
    """Sets the status of this job to 'queued' or 'waiting'."""
    # update the job id (i.e., when the job is executed in the grid)
    if new_job_id is not None:
      self.id = new_job_id

    if new_job_name is not None:
      self.name = new_job_name

    if queue_name is not None:
      self.queue_name = queue_name

    new_status = 'queued'
    self.result = None
    # check if we have to wait for another job to finish
    for job in self.get_jobs_we_wait_for():
      if job.status not in ('success', 'failure'):
        new_status = 'waiting'
      elif self.stop_on_failure and job.status == 'failure':
        new_status = 'failure'

    # reset the queued jobs that depend on us to waiting status
    for job in self.get_jobs_waiting_for_us():
      if job.status == 'queued':
        job.status = 'failure' if new_status == 'failure' else 'waiting'

    self.status = new_status
    for array_job in self.array:
      if array_job.status not in ('success', 'failure'):
        array_job.status = new_status