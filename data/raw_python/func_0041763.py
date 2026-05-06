def execute(self, array_id = None, machine_name = None):
    """Sets the status of this job to 'executing'."""
    self.status = 'executing'
    if array_id is not None:
      for array_job in self.array:
        if array_job.id == array_id:
          array_job.status = 'executing'
          if machine_name is not None:
            array_job.machine_name = machine_name
            array_job.start_time = datetime.now()
    elif machine_name is not None:
      self.machine_name = machine_name
    if self.start_time is None:
      self.start_time = datetime.now()

    # sometimes, the 'finish' command did not work for array jobs,
    # so check if any old job still has the 'executing' flag set
    for job in self.get_jobs_we_wait_for():
      if job.array and job.status == 'executing':
        job.finish(0, -1)