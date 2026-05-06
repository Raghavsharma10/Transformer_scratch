def server_factory(global_conf, host, port, **options):
  """Server factory for paste.

  Options are:
    * proactor: class name to use from cogen.core.proactors
      (default: DefaultProactor - best available proactor for current platform)
    * proactor_resolution: float
    * sched_default_priority: int (see cogen.core.util.priority)
    * sched_default_timeout: float (default: 0 - no timeout)
    * server_name: str
    * request_queue_size: int
    * sockoper_timeout: float (default: 15 - operations timeout in 15 seconds),
      -1 (no timeout), 0 (use scheduler's default), >0 (seconds)
    * sendfile_timeout: float (default: 300) - same as sockoper_timeout,
      only applied to sendfile operations (wich might need a much higher timeout
      value)
    * sockaccept_greedy: bool
  """
  port = int(port)

  try:
    import paste.util.threadinglocal as pastelocal
    pastelocal.local = local
  except ImportError:
    pass
  def serve(app):
    runner = Runner(host, port, app, options)
    runner.run()
  return serve