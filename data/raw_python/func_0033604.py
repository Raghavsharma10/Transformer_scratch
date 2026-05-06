def copy_file(self, from_path, to_path, replace=False):
        """
        Copies a file from a given source path to a destination path, adding
        appropriate parent directories when needed. If the destination path
        already exists and `replace` is `False`, the file will not be
        uploaded.
        """
        f = self.get_file(from_path)
        if f.copy(to_path, replace):
            self.set_cache_buster(to_path, f.hash())