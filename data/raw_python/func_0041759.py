def times(job):
  """Returns a string containing timing information for teh given job, which might be a :py:class:`Job` or an :py:class:`ArrayJob`."""
  timing = "Submitted: %s" % job.submit_time.ctime()
  if job.start_time is not None:
    timing += "\nStarted  : %s \t Job waited  : %s" % (job.start_time.ctime(), job.start_time - job.submit_time)
  if job.finish_time is not None:
    timing += "\nFinished : %s \t Job executed: %s" % (job.finish_time.ctime(), job.finish_time - job.start_time)
  return timing