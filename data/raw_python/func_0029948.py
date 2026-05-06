def build_ingest_fs(self):
        """Return a pyfilesystem subdirectory for the ingested source files"""

        base_path = 'ingest'

        if not self.build_fs.exists(base_path):
            self.build_fs.makedir(base_path, recursive=True, allow_recreate=True)

        return self.build_fs.opendir(base_path)