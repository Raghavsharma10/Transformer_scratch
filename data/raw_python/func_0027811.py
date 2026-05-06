def makeService(cls, options):
        """
        Create an L{IService} for the database specified by the given
        configuration.
        """
        from axiom.store import Store
        jm = options['journal-mode']
        if jm is not None:
            jm = jm.decode('ascii')
        store = Store(options['dbdir'], debug=options['debug'], journalMode=jm)
        service = IService(store)
        _CheckSystemVersion(store).setServiceParent(service)
        return service