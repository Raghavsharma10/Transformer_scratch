def sync_out(self, file_name=None, force=False):
        """Synchronize from objects to records"""
        self.log('---- Sync Out ----')
        from ambry.bundle.files import BuildSourceFile

        self.dstate = self.STATES.BUILDING

        for f in self.build_source_files.list_records():

            if (f.sync_dir() == BuildSourceFile.SYNC_DIR.RECORD_TO_FILE or f.record.path == file_name) or force:
                self.log('Sync: {}'.format(f.record.path))
                f.record_to_fs()

        self.commit()