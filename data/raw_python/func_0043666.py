def find_branches_raw(self):
        """Find information about the branches in the repository."""
        listing = self.context.capture('git', 'for-each-ref', '--format=%(refname)\t%(objectname)')
        for line in listing.splitlines():
            match = FOR_EACH_REF_PATTERN.match(line)
            if match and match.group('name') != 'HEAD':
                yield (match.group('prefix'),
                       match.group('name'),
                       match.group('revision_id'))