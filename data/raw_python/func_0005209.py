def parents(self):
        # type: () -> List[CommitDetails]
        """ Parents of the this commit. """
        if self._parents is None:
            self._parents = [CommitDetails.get(x) for x in self.parents_sha1]

        return self._parents