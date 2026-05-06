def acquire(self, **kwargs):
        """
        Copy the file and return its path

        Returns
        -------
        str or None
            The path of the file or None if it does not exist or if
            verification failed.
        """
        path = path_string(self.path)
        if os.path.exists(path):
            if config.verify_file(path, self.sha256):
                return path
        return None