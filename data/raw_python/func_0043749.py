def isseq(obj):
  '''
  Returns True if `obj` is a sequence-like object (but not a string or
  dict); i.e. a tuple, list, subclass thereof, or having an interface
  that supports iteration.
  '''
  return \
    not isstr(obj) \
    and not isdict(obj) \
    and ( isinstance(obj, (list, tuple)) \
          or callable(getattr(obj, '__iter__', None)))