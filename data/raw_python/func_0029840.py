def sync_dir(self):
        """ Report on which direction a synchronization should be done.
        :return:
        """

        # NOTE: These are ordered so the FILE_TO_RECORD has preference over RECORD_TO_FILE
        # if there is a conflict.

        if self.exists() and bool(self.size()) and not self.record.size:
            # The fs exists, but the record is empty
            return self.SYNC_DIR.FILE_TO_RECORD

        if (self.fs_modtime or 0) > (self.record.modified or 0) and self.record.source_hash != self.fs_hash:
            # Filesystem is newer

            return self.SYNC_DIR.FILE_TO_RECORD

        if self.record.size and not self.exists():
            # Record exists, but not the FS

            return self.SYNC_DIR.RECORD_TO_FILE

        if (self.record.modified or 0) > (self.fs_modtime or 0):
            # Record is newer
            return self.SYNC_DIR.RECORD_TO_FILE

        return None