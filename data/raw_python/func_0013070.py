def __get_arg(cls, kwds, kwd):
    """Internal helper method to parse keywords that may be property names."""
    alt_kwd = '_' + kwd
    if alt_kwd in kwds:
      return kwds.pop(alt_kwd)
    if kwd in kwds:
      obj = getattr(cls, kwd, None)
      if not isinstance(obj, Property) or isinstance(obj, ModelKey):
        return kwds.pop(kwd)
    return None