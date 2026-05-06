def submit(self, command_line, name = None, array = None, dependencies = [], exec_dir = None, log_dir = None, dry_run = False, stop_on_failure = False, **kwargs):
    """Submits a job that will be executed on the local machine during a call to "run".
    All kwargs will simply be ignored."""
    # remove duplicate dependencies
    dependencies = sorted(list(set(dependencies)))

    # add job to database
    self.lock()
    job = add_job(self.session, command_line=command_line, name=name, dependencies=dependencies, array=array, exec_dir=exec_dir, log_dir=log_dir, stop_on_failure=stop_on_failure)
    logger.info("Added job '%s' to the database", job)

    if dry_run:
      print("Would have added the Job", job, "to the database to be executed locally.")
      self.session.delete(job)
      logger.info("Deleted job '%s' from the database due to dry-run option", job)
      job_id = None
    else:
      job_id = job.unique

    # return the new job id
    self.unlock()
    return job_id