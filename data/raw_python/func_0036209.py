def execute_greenlet_async(func, *args, **kwargs):
  """
  Executes `func` in a separate greenlet in the same process. Memory and other
  resources are available (e.g. TCP connections etc.) `args` and `kwargs` are
  passed to `func`.
  """
  global _GREENLET_EXECUTOR
  if _GREENLET_EXECUTOR is None:
    _GREENLET_EXECUTOR = GreenletExecutor(
      num_greenlets=settings.node.greenlet_pool_size)
  return _GREENLET_EXECUTOR.submit(func, *args, **kwargs)