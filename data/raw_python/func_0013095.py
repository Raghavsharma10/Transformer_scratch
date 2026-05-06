def _is_default_hook(default_hook, hook):
    """Checks whether a specific hook is in its default state.

    Args:
      cls: A ndb.model.Model class.
      default_hook: Callable specified by ndb internally (do not override).
      hook: The hook defined by a model class using _post_*_hook.

    Raises:
      TypeError if either the default hook or the tested hook are not callable.
    """
    if not hasattr(default_hook, '__call__'):
      raise TypeError('Default hooks for ndb.model.Model must be callable')
    if not hasattr(hook, '__call__'):
      raise TypeError('Hooks must be callable')
    return default_hook.im_func is hook.im_func