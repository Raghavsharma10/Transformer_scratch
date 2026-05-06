def _wrap_method(name):
  """Wrap a method.

  Patch a method which might return a datetime.datetime to return a
  datetime_tz.datetime_tz instead.

  Args:
    name: The name of the method to patch
  """
  method = getattr(datetime.datetime, name)

  # Have to give the second argument as method has no __module__ option.
  @functools.wraps(method, ("__name__", "__doc__"), ())
  def wrapper(self, *args, **kw):
    r = method(self, *args, **kw)

    if isinstance(r, datetime.datetime) and not isinstance(r, type(self)):
      r = type(self)(r)
    return r

  setattr(datetime_tz, name, wrapper)