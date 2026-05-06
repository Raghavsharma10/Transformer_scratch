def stop_jobs(self, job_ids=None):
    """Resets the status of the job to 'submitted' when they are labeled as 'executing'."""
    self.lock()

    jobs = self.get_jobs(job_ids)
    for job in jobs:
      if job.status in ('executing', 'queued', 'waiting') and job.queue_name == 'local':
        logger.info("Reset job '%s' (%s) in the database", job.name, self._format_log(job.id))
        job.submit()

    self.session.commit()
    self.unlock()