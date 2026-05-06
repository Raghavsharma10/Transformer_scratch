def sort_members(tup, names):
  """Return two pairs of members, scalar and tuple members.

  The scalars will be sorted s.t. the unbound members are at the top.
  """
  scalars, tuples = partition(lambda x: not is_tuple_node(tup.member[x].value), names)
  unbound, bound = partition(lambda x: tup.member[x].value.is_unbound(), scalars)
  return usorted(unbound) + usorted(bound), usorted(tuples)