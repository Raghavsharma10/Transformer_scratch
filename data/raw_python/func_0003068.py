def process(self, checksum, revision=None):
        """
        Process a new revision and detect a revert if it occurred.  Note that
        you can pass whatever you like as `revision` and it will be returned in
        the case that a revert occurs.

        :Parameters:
            checksum : str
                Any identity-machable string-based hash of revision content
            revision : `mixed`
                Revision metadata.  Note that any data will just be returned
                in the case of a revert.

        :Returns:
            a :class:`~mwreverts.Revert` if one occured or `None`
        """
        revert = None

        if checksum in self:  # potential revert

            reverteds = list(self.up_to(checksum))

            if len(reverteds) > 0:  # If no reverted revisions, this is a noop
                revert = Revert(revision, reverteds, self[checksum])

        self.insert(checksum, revision)
        return revert