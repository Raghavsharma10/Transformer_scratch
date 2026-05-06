def wait(results, num=None, timeout=None):
  """
  Wait for results of async executions to become available/ready.

  `results`: List of AsyncResult instances returned by one of
             `execute_greenlet_async` or `execute_process_async`.
  `num`: Number of results to wait for. None implies wait for all results.
  `timeout`: Number of seconds to wait for `num` of the `results` to become
             ready.
  """
  return AbstractExecutor.wait(results, num=num, timeout=timeout)