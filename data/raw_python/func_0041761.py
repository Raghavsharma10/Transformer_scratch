def submit(self, new_queue = None):
    """Sets the status of this job to 'submitted'."""
    self.status = 'submitted'
    self.result = None
    self.machine_name = None
    if new_queue is not None:
      self.queue_name = new_queue
    for array_job in self.array:
      array_job.status = 'submitted'
      array_job.result = None
      array_job.machine_name = None
    self.submit_time = datetime.now()
    self.start_time = None
    self.finish_time = None