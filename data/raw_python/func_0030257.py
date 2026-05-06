def replace(self, **kw):
    """Return datetime with new specified fields given as arguments.

    For example, dt.replace(days=4) would return a new datetime_tz object with
    exactly the same as dt but with the days attribute equal to 4.

    Any attribute can be replaced, but tzinfo can not be set to None.

    Args:
      Any datetime_tz attribute.

    Returns:
      A datetime_tz object with the attributes replaced.

    Raises:
      TypeError: If the given replacement is invalid.
    """
    if "tzinfo" in kw:
      if kw["tzinfo"] is None:
        raise TypeError("Can not remove the timezone use asdatetime()")
      else:
        tzinfo = kw["tzinfo"]
        del kw["tzinfo"]
    else:
      tzinfo = None

    is_dst = None
    if "is_dst" in kw:
      is_dst = kw["is_dst"]
      del kw["is_dst"]
    else:
      # Use our own DST setting..
      is_dst = self.is_dst

    replaced = self.asdatetime().replace(**kw)

    return type(self)(
        replaced, tzinfo=tzinfo or self.tzinfo.zone, is_dst=is_dst)