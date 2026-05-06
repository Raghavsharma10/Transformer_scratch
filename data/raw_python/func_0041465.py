def resubmit(self, job_ids = None, also_success = False, running_jobs = False, new_command=None, verbosity=0, keep_logs=False, **kwargs):
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
        grid_status = qstat(job.id, context=self.context)
        if len(grid_status) != 0:
          logger.warn("Deleting job '%d' since it was still running in the grid." % job.unique)
          qdel(job.id, context=self.context)
        # re-submit job to the grid
        arguments = job.get_arguments()
        arguments.update(**kwargs)
        if ('queue' not in arguments or arguments['queue'] == 'all.q'):
          for arg in ('hvmem', 'pe_opt', 'io_big'):
            if arg in arguments:
              del arguments[arg]
        job.set_arguments(kwargs=arguments)
        # delete old status and result of the job
        if not keep_logs:
          self.delete_logs(job)
        job.submit()
        if job.queue_name == 'local' and 'queue' not in arguments:
          logger.warn("Re-submitting job '%s' locally (since no queue name is specified)." % job)
        else:
          deps = [dep.unique for dep in job.get_jobs_we_wait_for()]
          logger.debug("Re-submitting job '%s' with dependencies '%s' to the grid." % (job, deps))
          self._submit_to_grid(job, job.name, job.get_array(), deps, job.log_dir, verbosity, **arguments)

        # commit after each job to avoid failures of not finding the job during execution in the grid
        self.session.commit()
    self.unlock()