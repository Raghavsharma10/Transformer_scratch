def submit(self, command_line, name = None, array = None, dependencies = [], exec_dir = None, log_dir = "logs", dry_run = False, verbosity = 0, stop_on_failure = False, **kwargs):
    """Submits a job that will be executed in the grid."""
    # add job to database
    self.lock()
    job = add_job(self.session, command_line, name, dependencies, array, exec_dir=exec_dir, log_dir=log_dir, stop_on_failure=stop_on_failure, context=self.context, **kwargs)
    logger.info("Added job '%s' to the database." % job)
    if dry_run:
      print("Would have added the Job")
      print(job)
      print("to the database to be executed in the grid with options:", str(kwargs))
      self.session.delete(job)
      logger.info("Deleted job '%s' from the database due to dry-run option" % job)
      job_id = None

    else:
      job_id = self._submit_to_grid(job, name, array, dependencies, log_dir, verbosity, **kwargs)

    self.session.commit()
    self.unlock()

    return job_id