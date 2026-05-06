def resubmit(self, job_ids = None, also_success = False, running_jobs = False, new_command=None, keep_logs=False, **kwargs):
    """Re-submit jobs automatically"""
    self.lock()
    # iterate over all jobs
    jobs = self.get_jobs(job_ids)
    if new_command is not None:
      if len(jobs) == 1:
        jobs[0].set_command_line(new_command)
      else:
        logger.warn("Ignoring new command since no single job id was specified")
    accepted_old_status = ('submitted', 'success', 'failure') if also_success else ('submitted', 'failure',)
    for job in jobs:
      # check if this job needs re-submission
      if running_jobs or job.status in accepted_old_status:
        if job.queue_name != 'local' and job.status == 'executing':
          logger.error("Cannot re-submit job '%s' locally since it is still running in the grid. Use 'jman stop' to stop it\'s execution!", job)
        else:
          # re-submit job to the grid
          logger.info("Re-submitted job '%s' to the database", job)
          if not keep_logs:
            self.delete_logs(job)
          job.submit('local')

    self.session.commit()
    self.unlock()