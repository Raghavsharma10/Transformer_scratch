def get_thunk_env(self, k):
    """Return the thunk AND environment for validating it in for the given key.

    There might be different envs in case the thunk comes from a different (composed) tuple. If the thunk needs its
    environment bound on retrieval, that will be done here.
    """
    if k not in self.__items:
      raise exceptions.EvaluationError('Unknown key: %r in tuple %r' % (k, self))
    x = self.__items[k]

    env = self.env(self)

    # Bind this to the tuple's parent environment
    if isinstance(x, framework.BindableThunk):
      return x.bind(self.__parent_env), env
    return x, env