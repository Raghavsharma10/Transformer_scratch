def initRnaQuantificationSet(self):
        """
        Initialize an empty RNA quantification set
        """
        store = rnaseq2ga.RnaSqliteStore(self._args.filePath)
        store.createTables()