def _read_journal(self):
        """Extracts the USN journal from the disk and parses its content."""
        root = self._filesystem.inspect_get_roots()[0]
        inode = self._filesystem.stat('C:\\$Extend\\$UsnJrnl')['ino']

        with NamedTemporaryFile(buffering=0) as tempfile:
            self._filesystem.download_inode(root, inode, tempfile.name)

            journal = usn_journal(tempfile.name)

            return parse_journal(journal)