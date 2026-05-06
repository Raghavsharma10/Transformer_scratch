def get_member_node(self, key):
    """Return the AST node for the given member, from the first tuple that serves it."""
    for tup, _ in self.lookups:
      if key in tup:
        return tup.get_member_node(key)
    raise RuntimeError('Key not found in composite tuple: %r' % key)