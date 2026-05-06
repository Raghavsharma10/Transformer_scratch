def getFactory(self):
        """
        Create an L{AxiomSite} which supports authenticated and anonymous
        access.
        """
        checkers = [self.loginSystem, AllowAnonymousAccess()]
        guardedRoot = PersistentSessionWrapper(
            self.store,
            Portal(self.loginSystem, checkers),
            domains=[self.hostname])
        unguardedRoot = UnguardedWrapper(self.store, guardedRoot)
        securingRoot = SecuringWrapper(self, unguardedRoot)
        logPath = None
        if self.httpLog is not None:
            logPath = self.httpLog.path
        return AxiomSite(self.store, securingRoot, logPath=logPath)