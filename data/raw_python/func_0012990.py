def positional(max_pos_args):
  """A decorator to declare that only the first N arguments may be positional.

  Note that for methods, n includes 'self'.
  """
  __ndb_debug__ = 'SKIP'

  def positional_decorator(wrapped):
    if not DEBUG:
      return wrapped
    __ndb_debug__ = 'SKIP'

    @wrapping(wrapped)
    def positional_wrapper(*args, **kwds):
      __ndb_debug__ = 'SKIP'
      if len(args) > max_pos_args:
        plural_s = ''
        if max_pos_args != 1:
          plural_s = 's'
        raise TypeError(
            '%s() takes at most %d positional argument%s (%d given)' %
            (wrapped.__name__, max_pos_args, plural_s, len(args)))
      return wrapped(*args, **kwds)
    return positional_wrapper
  return positional_decorator