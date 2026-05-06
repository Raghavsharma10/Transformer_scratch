def find_tags(self):
        """
        Find information about the tags in the repository.

        .. note:: The ``bzr tags`` command reports tags pointing to
                  non-existing revisions as ``?`` but doesn't provide revision
                  ids. We can get the revision ids using the ``bzr tags
                  --show-ids`` command but this command doesn't mark tags
                  pointing to non-existing revisions. We combine the output of
                  both because we want all the information.
        """
        valid_tags = []
        listing = self.context.capture('bzr', 'tags')
        for line in listing.splitlines():
            tokens = line.split()
            if len(tokens) == 2 and tokens[1] != '?':
                valid_tags.append(tokens[0])
        listing = self.context.capture('bzr', 'tags', '--show-ids')
        for line in listing.splitlines():
            tokens = line.split()
            if len(tokens) == 2 and tokens[0] in valid_tags:
                tag, revision_id = tokens
                yield Revision(
                    repository=self,
                    revision_id=tokens[1],
                    tag=tokens[0],
                )