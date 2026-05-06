def uninstallFrom(self, target):
    """
    Remove this object from the target, as well as any dependencies
    that it automatically installed which were not explicitly
    "pinned" by calling "install", and raising an exception if
    anything still depends on this.
    """

    #did this class powerup on any interfaces? powerdown if so.
    target.powerDown(self)


    for dc in self.store.query(_DependencyConnector,
                               _DependencyConnector.target==target):
        if dc.installee is self:
            dc.deleteFromStore()

    for item in installedUniqueRequirements(self, target):
        uninstallFrom(item, target)

    callback = getattr(self, "uninstalled", None)
    if callback is not None:
        callback()