def _get_hierarchy(cls):
    """Internal helper to return the list of polymorphic base classes.

    This returns a list of class objects, e.g. [Animal, Feline, Cat].
    """
    bases = []
    for base in cls.mro():  # pragma: no branch
      if hasattr(base, '_get_hierarchy'):
        bases.append(base)
    del bases[-1]  # Delete PolyModel itself
    bases.reverse()
    return bases