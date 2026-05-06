def get_entity_group_version(key):
  """Return the version of the entity group containing key.

  Args:
    key: a key for an entity group whose __entity_group__ key you want.

  Returns:
    The version of the entity group containing key. This version is
    guaranteed to increase on every change to the entity group. The version
    may increase even in the absence of user-visible changes to the entity
    group. May return None if the entity group was never written to.

    On non-HR datatores, this function returns None.
  """

  eg = EntityGroup.key_for_entity_group(key).get()
  if eg:
    return eg.version
  else:
    return None