def report(self, job_ids=None, array_ids=None, output=True, error=True, status=Status, name=None):
    """Iterates through the output and error files and write the results to command line."""
    def _write_contents(job):
      # Writes the contents of the output and error files to command line
      out_file, err_file = job.std_out_file(), job.std_err_file()
      logger.info("Contents of output file: '%s'" % out_file)
      if output and out_file is not None and os.path.exists(out_file) and os.stat(out_file).st_size > 0:
        print(open(out_file).read().rstrip())
        print("-"*20)
      if error and err_file is not None and os.path.exists(err_file) and os.stat(err_file).st_size > 0:
        logger.info("Contents of error file: '%s'" % err_file)
        print(open(err_file).read().rstrip())
        print("-"*40)

    def _write_array_jobs(array_jobs):
      for array_job in array_jobs:
        print("Array Job", str(array_job.id), ("(%s) :"%array_job.machine_name if array_job.machine_name is not None else ":"))
        _write_contents(array_job)

    self.lock()

    # check if an array job should be reported
    if array_ids:
      if len(job_ids) != 1: logger.error("If array ids are specified exactly one job id must be given.")
      array_jobs = list(self.session.query(ArrayJob).join(Job).filter(Job.unique.in_(job_ids)).filter(Job.unique == ArrayJob.job_id).filter(ArrayJob.id.in_(array_ids)))
      if array_jobs: print(array_jobs[0].job)
      _write_array_jobs(array_jobs)

    else:
      # iterate over all jobs
      jobs = self.get_jobs(job_ids)
      for job in jobs:
        if name is not None and job.name != name:
          continue
        if job.status not in status:
          continue
        if job.array:
          print(job)
          _write_array_jobs(job.array)
        else:
          print(job)
          _write_contents(job)
        if job.log_dir is not None:
          print("-"*60)

    self.unlock()