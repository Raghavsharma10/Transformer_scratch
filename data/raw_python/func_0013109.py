def add_flow_exception(exc):
  """Add an exception that should not be logged.

  The argument must be a subclass of Exception.
  """
  global _flow_exceptions
  if not isinstance(exc, type) or not issubclass(exc, Exception):
    raise TypeError('Expected an Exception subclass, got %r' % (exc,))
  as_set = set(_flow_exceptions)
  as_set.add(exc)
  _flow_exceptions = tuple(as_set)