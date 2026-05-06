def find_tags(self):
        """Find information about the tags in the repository."""
        listing = self.context.capture('git', 'show-ref', '--tags', check=False)
        for line in listing.splitlines():
            tokens = line.split()
            if len(tokens) >= 2 and tokens[1].startswith('refs/tags/'):
                yield Revision(
                    repository=self,
                    revision_id=tokens[0],
                    tag=tokens[1][len('refs/tags/'):],
                )