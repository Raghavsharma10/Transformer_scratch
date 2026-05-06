def get_event_loop():
  """Return a EventLoop instance.

  A new instance is created for each new HTTP request.  We determine
  that we're in a new request by inspecting os.environ, which is reset
  at the start of each request.  Also, each thread gets its own loop.
  """
  ev = _state.event_loop
  if not os.getenv(_EVENT_LOOP_KEY) and ev is not None:
    ev.clear()
    _state.event_loop = None
    ev = None
  if ev is None:
    ev = EventLoop()
    _state.event_loop = ev
    os.environ[_EVENT_LOOP_KEY] = '1'
  return ev