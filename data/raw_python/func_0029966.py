def sync_in(self, force=False):
        """Synchronize from files to records, and records to objects"""
        self.log('---- Sync In ----')

        self.dstate = self.STATES.BUILDING

        for path_name in self.source_fs.listdir():

            f = self.build_source_files.instance_from_name(path_name)

            if not f:
                self.warn('Ignoring unknown file: {}'.format(path_name))
                continue

            if f and f.exists and (f.fs_is_newer or force):
                self.log('Sync: {}'.format(f.record.path))
                f.fs_to_record()
                f.record_to_objects()

        self.commit()

        self.library.search.index_bundle(self, force=True)