def execute_process_async(func, *args, **kwargs):
  """
  Executes `func` in a separate process. Memory and other resources are not
  available. This gives true concurrency at the cost of losing access to
  these resources. `args` and `kwargs` are
  """
  global _GIPC_EXECUTOR
  if _GIPC_EXECUTOR is None:
    _GIPC_EXECUTOR = GIPCExecutor(
      num_procs=settings.node.gipc_pool_size,
      num_greenlets=settings.node.greenlet_pool_size)
  return _GIPC_EXECUTOR.submit(func, *args, **kwargs)