def communicate(self, job_ids = None):
    """Communicates with the SGE grid (using qstat) to see if jobs are still running."""
    self.lock()
    # iterate over all jobs
    jobs = self.get_jobs(job_ids)
    for job in jobs:
      job.refresh()
      if job.status in ('queued', 'executing', 'waiting') and job.queue_name != 'local':
        status = qstat(job.id, context=self.context)
        if len(status) == 0:
          job.status = 'failure'
          job.result = 70 # ASCII: 'F'
          logger.warn("The job '%s' was not executed successfully (maybe a time-out happened). Please check the log files." % job)
          for array_job in job.array:
            if array_job.status in ('queued', 'executing'):
              array_job.status = 'failure'
              array_job.result = 70 # ASCII: 'F'


    self.session.commit()
    self.unlock()