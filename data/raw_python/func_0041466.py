def run_job(self, job_id, array_id = None):
    """Overwrites the run-job command from the manager to extract the correct job id before calling base class implementation."""
    # get the unique job id from the given grid id
    self.lock()
    jobs = list(self.session.query(Job).filter(Job.id == job_id))
    if len(jobs) != 1:
      self.unlock()
      raise ValueError("Could not find job id '%d' in the database'" % job_id)
    job_id = jobs[0].unique
    self.unlock()
    # call base class implementation with the corrected job id
    return JobManager.run_job(self, job_id, array_id)