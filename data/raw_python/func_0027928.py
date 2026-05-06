def createNew(cls, store, pathSegments):
        """
        Create a new SubStore, allocating a new file space for it.
        """
        if isinstance(pathSegments, basestring):
            raise ValueError(
                'Received %r instead of a sequence' % (pathSegments,))
        if store.dbdir is None:
            self = cls(store=store, storepath=None)
        else:
            storepath = store.newDirectory(*pathSegments)
            self = cls(store=store, storepath=storepath)
        self.open()
        self.close()
        return self