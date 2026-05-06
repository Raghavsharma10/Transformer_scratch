def sync(self, force=None):
        """Synchronize between the file in the file system and the field record"""

        try:
            if force:
                sd = force
            else:
                sd = self.sync_dir()

            if sd == self.SYNC_DIR.FILE_TO_RECORD:

                if force and not self.exists():
                    return None

                self.fs_to_record()

            elif sd == self.SYNC_DIR.RECORD_TO_FILE:
                self.record_to_fs()

            else:
                return None

            self._dataset.config.sync[self.file_const][sd] = time.time()
            return sd
        except Exception as e:
            self._bundle.rollback()
            self._bundle.error("Failed to sync '{}': {}".format(self.file_const, e))
            raise