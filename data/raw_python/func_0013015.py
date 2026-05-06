def default_memcache_timeout_policy(key):
    """Default memcache timeout policy.

    This defers to _memcache_timeout on the Model class.

    Args:
      key: Key instance.

    Returns:
      Memcache timeout to use (integer), or None.
    """
    timeout = None
    if key is not None and isinstance(key, model.Key):
      modelclass = model.Model._kind_map.get(key.kind())
      if modelclass is not None:
        policy = getattr(modelclass, '_memcache_timeout', None)
        if policy is not None:
          if isinstance(policy, (int, long)):
            timeout = policy
          else:
            timeout = policy(key)
    return timeout