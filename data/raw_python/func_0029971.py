def sync_objects_out(self, force=False):
        """Synchronize from objects to records, and records to files"""
        self.log('---- Sync Objects Out ----')
        from ambry.bundle.files import BuildSourceFile

        self.dstate = self.STATES.BUILDING

        for f in self.build_source_files.list_records():

            self.log('Sync: {}'.format(f.record.path))
            f.objects_to_record()

        self.commit()