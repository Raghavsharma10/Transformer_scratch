def installedUniqueRequirements(self, target):
    """
    Return an iterable of things installed on the target that this item
    requires and are not required by anything else.
    """

    myDepends = dependentsOf(self.__class__)
    #XXX optimize?
    for dc in self.store.query(_DependencyConnector,
                               _DependencyConnector.target==target):
        if dc.installee is self:
            #we're checking all the others not ourself
            continue
        depends = dependentsOf(dc.installee.__class__)
        if self.__class__ in depends:
            raise DependencyError(
                "%r cannot be uninstalled from %r, "
                "%r still depends on it" % (self, target, dc.installee))

        for cls in myDepends[:]:
            #If one of my dependencies is required by somebody
            #else, leave it alone
            if cls in depends:
                myDepends.remove(cls)

    for dc in self.store.query(_DependencyConnector,
                               _DependencyConnector.target==target):
        if (dc.installee.__class__ in myDepends
            and not dc.explicitlyInstalled):
            yield dc.installee