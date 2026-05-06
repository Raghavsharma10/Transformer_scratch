def get_namespaces(start=None, end=None):
  """Return all namespaces in the specified range.

  Args:
    start: only return namespaces >= start if start is not None.
    end: only return namespaces < end if end is not None.

  Returns:
    A list of namespace names between the (optional) start and end values.
  """
  q = Namespace.query()
  if start is not None:
    q = q.filter(Namespace.key >= Namespace.key_for_namespace(start))
  if end is not None:
    q = q.filter(Namespace.key < Namespace.key_for_namespace(end))
  return [x.namespace_name for x in q]