def stop_job(self, job_id, array_id = None):
    """Resets the status of the given to 'submitted' when they are labeled as 'executing'."""
    self.lock()

    job, array_job = self._job_and_array(job_id, array_id)
    if job is not None:
      if job.status in ('executing', 'queued', 'waiting'):
        logger.info("Reset job '%s' (%s) in the database", job.name, self._format_log(job.id))
        job.status = 'submitted'

      if array_job is not None and array_job.status in ('executing', 'queued', 'waiting'):
        logger.debug("Reset array job '%s' in the database", array_job)
        array_job.status = 'submitted'
      if array_job is None:
        for array_job in job.array:
          if array_job.status in ('executing', 'queued', 'waiting'):
            logger.debug("Reset array job '%s' in the database", array_job)
            array_job.status = 'submitted'

    self.session.commit()
    self.unlock()