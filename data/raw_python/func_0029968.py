def sync_in_records(self, force=False):
        """Synchronize from files to records"""
        self.log('---- Sync Files ----')

        for f in self.build_source_files:
            f.record_to_objects()

        # Only the metadata needs to be driven to the objects, since the other files are used as code,
        # directly from the file record.
        self.build_source_files.file(File.BSFILE.META).record_to_objects()

        self.commit()