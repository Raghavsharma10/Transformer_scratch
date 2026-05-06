def installedDependents(self, target):
    """
    Return an iterable of things installed on the target that
    require this item.
    """
    for dc in self.store.query(_DependencyConnector,
                               _DependencyConnector.target == target):
        depends = dependentsOf(dc.installee.__class__)
        if self.__class__ in depends:
            yield dc.installee