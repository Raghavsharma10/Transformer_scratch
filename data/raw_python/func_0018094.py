def env(self, current_scope):
    """Return an environment that will look up in current_scope for keys in
    this tuple, and the parent env otherwise.
    """
    return self.__env_cache.get(
            current_scope.ident,
            framework.Environment, current_scope,
            names=self.keys(),
            parent=framework.Environment({'self': current_scope}, parent=self.__parent_env))