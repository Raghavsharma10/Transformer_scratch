def delete(self, job_ids, array_ids = None, delete_logs = True, delete_log_dir = False, status = Status, delete_jobs = True):
    """Deletes the jobs with the given ids from the database."""
    def _delete_dir_if_empty(log_dir):
      if log_dir and delete_log_dir and os.path.isdir(log_dir) and not os.listdir(log_dir):
        os.rmdir(log_dir)
        logger.info("Removed empty log directory '%s'" % log_dir)

    def _delete(job, try_to_delete_dir=False):
      # delete the job from the database
      if delete_logs:
        self.delete_logs(job)
        if try_to_delete_dir:
          _delete_dir_if_empty(job.log_dir)
      if delete_jobs:
        self.session.delete(job)


    self.lock()

    # check if array ids are specified
    if array_ids:
      if len(job_ids) != 1: logger.error("If array ids are specified exactly one job id must be given.")
      array_jobs = list(self.session.query(ArrayJob).join(Job).filter(Job.unique.in_(job_ids)).filter(Job.unique == ArrayJob.job_id).filter(ArrayJob.id.in_(array_ids)))
      if array_jobs:
        job = array_jobs[0].job
        for array_job in array_jobs:
          if array_job.status in status:
            if delete_jobs:
              logger.debug("Deleting array job '%d' of job '%d' from the database." % (array_job.id, job.unique))
            _delete(array_job)
        if not job.array:
          if job.status in status:
            if delete_jobs:
              logger.info("Deleting job '%d' from the database." % job.unique)
            _delete(job, delete_jobs)

    else:
      # iterate over all jobs
      jobs = self.get_jobs(job_ids)
      for job in jobs:
        # delete all array jobs
        if job.array:
          for array_job in job.array:
            if array_job.status in status:
              if delete_jobs:
                logger.debug("Deleting array job '%d' of job '%d' from the database." % (array_job.id, job.unique))
              _delete(array_job)
        # delete this job
        if job.status in status:
          if delete_jobs:
            logger.info("Deleting job '%d' from the database." % job.unique)
          _delete(job, delete_jobs)

    self.session.commit()
    self.unlock()