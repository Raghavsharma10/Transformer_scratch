def tobool(obj, default=False):
  '''
  Returns a bool representation of `obj`: if `obj` is not a string,
  it is returned cast to a boolean by calling `bool()`. Otherwise, it
  is checked for "truthy" or "falsy" values, and that is returned. If
  it is not truthy or falsy, `default` is returned (which defaults to
  ``False``) unless `default` is set to ``ValueError``, in which case
  an exception is raised.

  '''
  if isinstance(obj, bool):
    return obj
  if not isstr(obj):
    return bool(obj)
  lobj = obj.lower()
  if lobj in truthy:
    return True
  if lobj in falsy:
    return False
  if default is ValueError:
    raise ValueError('invalid literal for tobool(): %r' % (obj,))
  return default