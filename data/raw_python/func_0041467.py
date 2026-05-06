def stop_jobs(self, job_ids):
    """Stops the jobs in the grid."""
    self.lock()

    jobs = self.get_jobs(job_ids)
    for job in jobs:
      if job.status in ('executing', 'queued', 'waiting'):
        qdel(job.id, context=self.context)
        logger.info("Stopped job '%s' in the SGE grid." % job)
        job.submit()

      self.session.commit()
    self.unlock()