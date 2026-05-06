def queue_call(self, delay, callback, *args, **kwds):
    """Schedule a function call at a specific time in the future."""
    if delay is None:
      self.current.append((callback, args, kwds))
      return
    if delay < 1e9:
      when = delay + self.clock.now()
    else:
      # Times over a billion seconds are assumed to be absolute.
      when = delay
    self.insort_event_right((when, callback, args, kwds))