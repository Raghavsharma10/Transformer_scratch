def _gql(query_string, query_class=Query):
  """Parse a GQL query string (internal version).

  Args:
    query_string: Full GQL query, e.g. 'SELECT * FROM Kind WHERE prop = 1'.
    query_class: Optional class to use, default Query.

  Returns:
    An instance of query_class.
  """
  from .google_imports import gql  # Late import, to avoid name conflict.
  gql_qry = gql.GQL(query_string)
  kind = gql_qry.kind()
  if kind is None:
    # The query must be lacking a "FROM <kind>" class.  Let Expando
    # stand in for the model class (it won't actually be used to
    # construct the results).
    modelclass = model.Expando
  else:
    modelclass = model.Model._lookup_model(
        kind,
        tasklets.get_context()._conn.adapter.default_model)
    # Adjust kind to the kind of the model class.
    kind = modelclass._get_kind()
  ancestor = None
  flt = gql_qry.filters()
  filters = list(modelclass._default_filters())
  for name_op in sorted(flt):
    name, op = name_op
    values = flt[name_op]
    op = op.lower()
    if op == 'is' and name == gql.GQL._GQL__ANCESTOR:
      if len(values) != 1:
        raise ValueError('"is" requires exactly one value')
      [(func, args)] = values
      ancestor = _args_to_val(func, args)
      continue
    if op not in _OPS:
      raise NotImplementedError('Operation %r is not supported.' % op)
    for (func, args) in values:
      val = _args_to_val(func, args)
      prop = _get_prop_from_modelclass(modelclass, name)
      if prop._name != name:
        raise RuntimeError('Whoa! _get_prop_from_modelclass(%s, %r) '
                           'returned a property whose name is %r?!' %
                           (modelclass.__name__, name, prop._name))
      if isinstance(val, ParameterizedThing):
        node = ParameterNode(prop, op, val)
      elif op == 'in':
        node = prop._IN(val)
      else:
        node = prop._comparison(op, val)
      filters.append(node)
  if filters:
    filters = ConjunctionNode(*filters)
  else:
    filters = None
  orders = _orderings_to_orders(gql_qry.orderings(), modelclass)
  offset = gql_qry.offset()
  limit = gql_qry.limit()
  if limit < 0:
    limit = None
  keys_only = gql_qry._keys_only
  if not keys_only:
    keys_only = None
  options = QueryOptions(offset=offset, limit=limit, keys_only=keys_only)
  projection = gql_qry.projection()
  if gql_qry.is_distinct():
    group_by = projection
  else:
    group_by = None
  qry = query_class(kind=kind,
                    ancestor=ancestor,
                    filters=filters,
                    orders=orders,
                    default_options=options,
                    projection=projection,
                    group_by=group_by)
  return qry