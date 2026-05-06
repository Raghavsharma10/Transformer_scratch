def sync_sources(self, force=False):
        """Sync in only the sources.csv file"""
        from ambry.orm.file import File

        self.dstate = self.STATES.BUILDING

        synced = 0

        for fc in [File.BSFILE.SOURCES]:
            bsf = self.build_source_files.file(fc)
            if bsf.fs_is_newer or force:
                self.log('Syncing {}'.format(bsf.file_name))
                bsf.fs_to_objects()
                synced += 1

        return synced