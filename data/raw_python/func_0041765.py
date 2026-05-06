def refresh(self):
    """Refreshes the status information."""
    if self.status == 'executing' and self.array:
      new_result = 0
      for array_job in self.array:
        if array_job.status == 'failure' and new_result is not None:
          new_result = array_job.result
        elif array_job.status not in ('success', 'failure'):
          new_result = None
      if new_result is not None:
        self.status = 'success' if new_result == 0 else 'failure'
        self.result = new_result