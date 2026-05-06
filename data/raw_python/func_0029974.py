def sync_schema(self):
        """Sync in code files and the meta file, avoiding syncing the larger files"""
        from ambry.orm.file import File
        from ambry.bundle.files import BuildSourceFile

        self.dstate = self.STATES.BUILDING

        synced = 0
        for fc in [File.BSFILE.SCHEMA, File.BSFILE.SOURCESCHEMA]:
            bsf = self.build_source_files.file(fc)
            if bsf.fs_is_newer:
                self.log('Syncing {}'.format(bsf.file_name))
                bsf.sync(BuildSourceFile.SYNC_DIR.FILE_TO_RECORD)
                synced += 1

        return synced