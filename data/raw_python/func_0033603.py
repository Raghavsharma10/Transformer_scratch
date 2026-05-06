def upload_file(self, path, contents, replace=False):
        """
        Uplodas the file to its path with the given `content`, adding the
        appropriate parent directories when needed. If the path already exists
        and `replace` is `False`, the file will not be uploaded.
        """
        f = self.get_file(path)
        f.upload(contents, replace=replace)
        self.set_cache_buster(path, f.hash())