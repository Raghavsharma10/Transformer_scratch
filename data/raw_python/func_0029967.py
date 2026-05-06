def sync_in_files(self, force=False):
        """Synchronize from files to records"""
        self.log('---- Sync Files ----')

        self.dstate = self.STATES.BUILDING

        for f in self.build_source_files:

            if self.source_fs.exists(f.record.path):
                # print f.path, f.fs_modtime, f.record.modified, f.record.source_hash, f.fs_hash
                if f.fs_is_newer or force:
                    self.log('Sync: {}'.format(f.record.path))
                    f.fs_to_record()

        self.commit()