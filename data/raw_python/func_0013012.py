def default_datastore_policy(key):
    """Default datastore policy.

    This defers to _use_datastore on the Model class.

    Args:
      key: Key instance.

    Returns:
      A bool or None.
    """
    flag = None
    if key is not None:
      modelclass = model.Model._kind_map.get(key.kind())
      if modelclass is not None:
        policy = getattr(modelclass, '_use_datastore', None)
        if policy is not None:
          if isinstance(policy, bool):
            flag = policy
          else:
            flag = policy(key)
    return flag