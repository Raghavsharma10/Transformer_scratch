def finish(self, result, array_id = None):
    """Sets the status of this job to 'success' or 'failure'."""
    # check if there is any array job still running
    new_status = 'success' if result == 0 else 'failure'
    new_result = result
    finished = True
    if array_id is not None:
      for array_job in self.array:
        if array_job.id == array_id:
          array_job.status = new_status
          array_job.result = result
          array_job.finish_time = datetime.now()
        if array_job.status not in ('success', 'failure'):
          finished = False
        elif new_result == 0:
          new_result = array_job.result

    if finished:
      # There was no array job, or all array jobs finished
      self.status = 'success' if new_result == 0 else 'failure'
      self.result = new_result
      self.finish_time = datetime.now()

      # update all waiting jobs
      for job in self.get_jobs_waiting_for_us():
        if job.status == 'waiting':
          job.queue()