def _getStore(self):
        """
        Get the Store used for FTS.

        If it does not exist, it is created and initialised.
        """
        storeDir = self.store.newDirectory(self.indexDirectory)
        if not storeDir.exists():
            store = Store(storeDir)
            self._initStore(store)
            return store
        else:
            return Store(storeDir)