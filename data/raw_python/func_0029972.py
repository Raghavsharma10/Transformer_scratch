def sync_code(self):
        """Sync in code files and the meta file, avoiding syncing the larger files"""
        from ambry.orm.file import File
        from ambry.bundle.files import BuildSourceFile

        self.dstate = self.STATES.BUILDING

        synced = 0

        for fc in [File.BSFILE.BUILD, File.BSFILE.META, File.BSFILE.LIB, File.BSFILE.TEST, File.BSFILE.DOC]:
            bsf = self.build_source_files.file(fc)
            if bsf.fs_is_newer:
                self.log('Syncing {}'.format(bsf.file_name))
                bsf.sync(BuildSourceFile.SYNC_DIR.FILE_TO_RECORD)
                synced += 1

        # Only the metadata needs to be driven to the objects, since the other files are used as code,
        # directly from the file record.
        self.build_source_files.file(File.BSFILE.META).record_to_objects()

        return synced