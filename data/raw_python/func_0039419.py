def _visit_filesystem(self):
        """Walks through the filesystem content."""
        self.logger.debug("Parsing File System content.")

        root_partition = self._filesystem.inspect_get_roots()[0]

        yield from self._root_dirent()

        for entry in self._filesystem.filesystem_walk(root_partition):
            yield Dirent(
                entry['tsk_inode'],
                self._filesystem.path('/' + entry['tsk_name']),
                entry['tsk_size'], entry['tsk_type'],
                True if entry['tsk_flags'] & TSK_ALLOC else False,
                timestamp(entry['tsk_atime_sec'], entry['tsk_atime_nsec']),
                timestamp(entry['tsk_mtime_sec'], entry['tsk_mtime_nsec']),
                timestamp(entry['tsk_ctime_sec'], entry['tsk_ctime_nsec']),
                timestamp(entry['tsk_crtime_sec'], entry['tsk_crtime_nsec']))