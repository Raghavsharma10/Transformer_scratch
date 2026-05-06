def find_branches(self):
        """
        Find the branches in the Mercurial repository.

        :returns: A generator of :class:`.Revision` objects.

        .. note:: Closed branches are not included.
        """
        listing = self.context.capture('hg', 'branches')
        for line in listing.splitlines():
            tokens = line.split()
            if len(tokens) >= 2 and ':' in tokens[1]:
                revision_number, revision_id = tokens[1].split(':')
                yield Revision(
                    branch=tokens[0],
                    repository=self,
                    revision_id=revision_id,
                    revision_number=int(revision_number),
                )