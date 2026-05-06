def usnjrnl_timeline(self):
        """Iterates over the changes occurred within the filesystem.

        Yields UsnJrnlEvent namedtuples containing:

            file_reference_number: known in Unix FS as inode.
            path: full path of the file.
            size: size of the file in bytes if recoverable.
            allocated: whether the file exists or it has been deleted.
            timestamp: timespamp of the change.
            changes: list of changes applied to the file.
            attributes: list of file attributes.

        """
        filesystem_content = defaultdict(list)

        self.logger.debug("Extracting Update Sequence Number journal.")

        journal = self._read_journal()

        for dirent in self._visit_filesystem():
            filesystem_content[dirent.inode].append(dirent)

        self.logger.debug("Generating timeline.")
        yield from generate_timeline(journal, filesystem_content)