def is_identifier_position(rootpath):
  """Return whether the cursor is in identifier-position in a member declaration."""
  if len(rootpath) >= 2 and is_tuple_member_node(rootpath[-2]) and is_identifier(rootpath[-1]):
    return True
  if len(rootpath) >= 1 and is_tuple_node(rootpath[-1]):
    # No deeper node than tuple? Must be identifier position, otherwise we'd have a TupleMemberNode.
    return True
  return False