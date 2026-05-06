def _analyze_indexed_fields(indexed_fields):
  """Internal helper to check a list of indexed fields.

  Args:
    indexed_fields: A list of names, possibly dotted names.

  (A dotted name is a string containing names separated by dots,
  e.g. 'foo.bar.baz'.  An undotted name is a string containing no
  dots, e.g. 'foo'.)

  Returns:
    A dict whose keys are undotted names.  For each undotted name in
    the argument, the dict contains that undotted name as a key with
    None as a value.  For each dotted name in the argument, the dict
    contains the first component as a key with a list of remainders as
    values.

  Example:
    If the argument is ['foo.bar.baz', 'bar', 'foo.bletch'], the return
    value is {'foo': ['bar.baz', 'bletch'], 'bar': None}.

  Raises:
    TypeError if an argument is not a string.
    ValueError for duplicate arguments and for conflicting arguments
      (when an undotted name also appears as the first component of
      a dotted name).
  """
  result = {}
  for field_name in indexed_fields:
    if not isinstance(field_name, basestring):
      raise TypeError('Field names must be strings; got %r' % (field_name,))
    if '.' not in field_name:
      if field_name in result:
        raise ValueError('Duplicate field name %s' % field_name)
      result[field_name] = None
    else:
      head, tail = field_name.split('.', 1)
      if head not in result:
        result[head] = [tail]
      elif result[head] is None:
        raise ValueError('Field name %s conflicts with ancestor %s' %
                         (field_name, head))
      else:
        result[head].append(tail)
  return result