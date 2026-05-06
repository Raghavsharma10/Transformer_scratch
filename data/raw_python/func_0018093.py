def get_no_validate(self, key):
    """Return an item without validating the schema."""
    x, env = self.get_thunk_env(key)

    # Check if this is a Thunk that needs to be lazily evaluated before we
    # return it.
    if isinstance(x, framework.Thunk):
      x = framework.eval(x, env)

    return x