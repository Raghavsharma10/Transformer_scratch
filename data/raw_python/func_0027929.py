def createStore(self, debug, journalMode=None):
        """
        Create the actual Store this Substore represents.
        """
        if self.storepath is None:
            self.store._memorySubstores.append(self) # don't fall out of cache
            if self.store.filesdir is None:
                filesdir = None
            else:
                filesdir = (self.store.filesdir.child("_substore_files")
                                               .child(str(self.storeID))
                                               .path)
            return Store(parent=self.store,
                         filesdir=filesdir,
                         idInParent=self.storeID,
                         debug=debug,
                         journalMode=journalMode)
        else:
            return Store(self.storepath.path,
                         parent=self.store,
                         idInParent=self.storeID,
                         debug=debug,
                         journalMode=journalMode)