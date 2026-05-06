def acquire(self, **kwargs):
        """
        Download the file and return its path

        Returns
        -------
        str or None
            The path of the file in BatchUp's temporary directory or None if
            the download failed.
        """
        return config.download_data(self.temp_filename, self.url,
                                    self.sha256)