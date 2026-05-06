def installedRequirements(self, target):
    """
    Return an iterable of things installed on the target that this
    item requires.
    """
    myDepends = dependentsOf(self.__class__)
    for dc in self.store.query(_DependencyConnector,
                               _DependencyConnector.target == target):
        if dc.installee.__class__ in myDepends:
            yield dc.installee