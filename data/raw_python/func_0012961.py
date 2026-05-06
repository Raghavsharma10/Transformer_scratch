def run_idle(self):
    """Run one of the idle callbacks.

    Returns:
      True if one was called, False if no idle callback was called.
    """
    if not self.idlers or self.inactive >= len(self.idlers):
      return False
    idler = self.idlers.popleft()
    callback, args, kwds = idler
    _logging_debug('idler: %s', callback.__name__)
    res = callback(*args, **kwds)
    # See add_idle() for the meaning of the callback return value.
    if res is not None:
      if res:
        self.inactive = 0
      else:
        self.inactive += 1
      self.idlers.append(idler)
    else:
      _logging_debug('idler %s removed', callback.__name__)
    return True