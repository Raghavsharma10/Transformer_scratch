def _reset_kind_map(cls):
    """Clear the kind map.  Useful for testing."""
    # Preserve "system" kinds, like __namespace__
    keep = {}
    for name, value in cls._kind_map.iteritems():
      if name.startswith('__') and name.endswith('__'):
        keep[name] = value
    cls._kind_map.clear()
    cls._kind_map.update(keep)