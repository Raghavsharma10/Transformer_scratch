def _query(cls, *args, **kwds):
    """Create a Query object for this class.

    Args:
      distinct: Optional bool, short hand for group_by = projection.
      *args: Used to apply an initial filter
      **kwds: are passed to the Query() constructor.

    Returns:
      A Query object.
    """
    # Validating distinct.
    if 'distinct' in kwds:
      if 'group_by' in kwds:
        raise TypeError(
            'cannot use distinct= and group_by= at the same time')
      projection = kwds.get('projection')
      if not projection:
        raise TypeError(
            'cannot use distinct= without projection=')
      if kwds.pop('distinct'):
        kwds['group_by'] = projection

    # TODO: Disallow non-empty args and filter=.
    from .query import Query  # Import late to avoid circular imports.
    qry = Query(kind=cls._get_kind(), **kwds)
    qry = qry.filter(*cls._default_filters())
    qry = qry.filter(*args)
    return qry