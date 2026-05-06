def get_node(self, key):
    """Delegate to our current "value provider" for the node belonging to this key."""
    if key in self.names:
      return self.values.get_member_node(key) if hasattr(self.values, 'get_member_node') else None
    return self.parent.get_node(key)