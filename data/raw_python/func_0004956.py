def get_subpath(self, subpath: str):
        """Search a file or directory relative to the base path"""
        for d in self._path:
            if os.path.exists(os.path.join(d, subpath)):
                return os.path.join(d, subpath)
        raise FileNotFoundError