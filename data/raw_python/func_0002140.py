def download(self, filename=None):
        """Download the dataset to a local file.

        Parameters
        ----------
        filename : str, optional
            The full path to which the dataset will be saved

        """
        if filename is None:
            filename = self.name
        with self.remote_open() as infile:
            with open(filename, 'wb') as outfile:
                outfile.write(infile.read())