def installedOn(self):
    """
    If this item is installed on another item, return the install
    target. Otherwise return None.
    """
    try:
        return self.store.findUnique(_DependencyConnector,
                                     _DependencyConnector.installee == self
                                     ).target
    except ItemNotFound:
        return None