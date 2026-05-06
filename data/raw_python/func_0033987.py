def _rel_path(self, path, basepath=None):
        """
        trim off basepath
        """
        basepath = basepath or self.src_dir
        return path[len(basepath) + 1:]