def reset(self):
        """
        Process everything all over again.
        """
        self.indexCount = 0
        indexDir = self.store.newDirectory(self.indexDirectory)
        if indexDir.exists():
            indexDir.remove()
        for src in self.getSources():
            src.removeReliableListener(self)
            src.addReliableListener(self, style=iaxiom.REMOTE)