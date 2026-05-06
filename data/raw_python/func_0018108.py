def flatten(list_of_lists):
  """flatten([[A]]) -> [A]

  Flatten a list of lists.
  """
  ret = []
  for lst in list_of_lists:
    if not isinstance(lst, list):
      raise ValueError('%r is not a list' % lst)
    ret.extend(lst)
  return ret