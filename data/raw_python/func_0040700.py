def reduce_contexts(parent, local):

  """Combine two test contexts into one.
  For value types of dict and list, the new context will aggregate the parent
  and local contexts. For other types, the value of the local context will
  replace the value of the parent (if any)."""

  context = {}

  for k,v in parent.items():
    if type(v) == dict:
      d = v.copy()
      d.update(local.get(k,{}))
      context[k] = d
    elif type(v) == list:
      context[k] = v + ensure_list(local.get(k,[]))
    else:
      context[k] = local.get(k,v)

  for k in set(local.keys()) - set(parent.keys()):
    context[k] = local[k]

  return context