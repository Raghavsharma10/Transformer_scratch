def merge_conflicts(self):
        """The filenames of any files with merge conflicts (a list of strings)."""
        filenames = set()
        listing = self.context.capture('git', 'ls-files', '--unmerged', '-z')
        for entry in split(listing, '\0'):
            # The output of `git ls-files --unmerged -z' consists of two
            # tab-delimited fields per zero-byte terminated record, where the
            # first field contains metadata and the second field contains the
            # filename. A single filename can be output more than once.
            metadata, _, name = entry.partition('\t')
            if metadata and name:
                filenames.add(name)
        return sorted(filenames)