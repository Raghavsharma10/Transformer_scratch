def gql(query_string, *args, **kwds):
  """Parse a GQL query string.

  Args:
    query_string: Full GQL query, e.g. 'SELECT * FROM Kind WHERE prop = 1'.
    *args, **kwds: If present, used to call bind().

  Returns:
    An instance of query_class.
  """
  qry = _gql(query_string)
  if args or kwds:
    qry = qry._bind(args, kwds)
  return qry