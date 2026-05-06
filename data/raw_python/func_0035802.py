def _validate_and_get_value(options, options_name, key, _type):
  """
  Check that `options` has a value for `key` with type
  `_type`. Return that value. `options_name` is a string representing a
  human-readable name for `options` to be used when printing errors.
  """
  if isinstance(options, dict):
    has = lambda k: k in options
    get = lambda k: options[k]
  elif isinstance(options, object):
    has = lambda k: hasattr(options, k)
    get = lambda k: getattr(options, k)
  else:
    raise ImproperlyConfigured(
        '`{}` must be a dictionary-like object.'.format(options_name))

  if not has(key):
    raise ImproperlyConfigured(
        '`{}` must be specified in `{}`'.format(key, options_name))

  value = get(key)
  if not isinstance(value, _type):
    raise ImproperlyConfigured(
        '`{}` in `{}` must be a {}'.format(key, options_name, repr(_type)))

  return value