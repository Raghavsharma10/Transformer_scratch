def removeReferenceSet(self):
        """
        Removes a referenceSet from the repo.
        """
        self._openRepo()
        referenceSet = self._repo.getReferenceSetByName(
            self._args.referenceSetName)

        def func():
            self._updateRepo(self._repo.removeReferenceSet, referenceSet)
        self._confirmDelete("ReferenceSet", referenceSet.getLocalId(), func)