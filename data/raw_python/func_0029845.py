def fh_to_record(self, f):
        """Load a file in the filesystem into the file record"""

        fn_path = self.file_name
        fr = self.record
        fr.path = fn_path

        fr.update_contents(f.read(), 'text/plain')

        fr.source_hash = self.fs_hash
        fr.synced_fs = self.fs_modtime
        fr.modified = self.fs_modtime