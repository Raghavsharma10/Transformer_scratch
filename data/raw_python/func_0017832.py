def acquire(self, **kwargs):
        """
        Copy the file and return its path

        Returns
        -------
        str or None
            The path of the file in BatchUp's temporary directory or None if
            the copy failed.
        """
        if self.source_path is None:
            source_path = kwargs[self.arg_name]
        else:
            source_path = self.source_path
        return config.copy_data(self.temp_filename, source_path, self.sha256)