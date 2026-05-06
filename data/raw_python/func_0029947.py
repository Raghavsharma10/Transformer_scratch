def build_partition_fs(self):
        """Return a pyfilesystem subdirectory for the build directory for the bundle. This the sub-directory
        of the build FS that holds the compiled SQLite file and the partition data files"""

        base_path = os.path.dirname(self.identity.cache_key)

        if not self.build_fs.exists(base_path):
            self.build_fs.makedir(base_path, recursive=True, allow_recreate=True)

        return self.build_fs.opendir(base_path)