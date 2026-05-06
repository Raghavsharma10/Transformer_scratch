def find_tags(self):
        """Find information about the tags in the repository."""
        listing = self.context.capture('hg', 'tags')
        for line in listing.splitlines():
            tokens = line.split()
            if len(tokens) >= 2 and ':' in tokens[1]:
                revision_number, revision_id = tokens[1].split(':')
                yield Revision(
                    repository=self,
                    revision_id=revision_id,
                    revision_number=int(revision_number),
                    tag=tokens[0],
                )