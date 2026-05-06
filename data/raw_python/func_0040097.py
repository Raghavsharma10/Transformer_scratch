def fpath(self, version=None, tags=None, ext=None):
        """Returns the filepath appropriate for an instance of this dataset.

        Parameters
        ----------
        version: str, optional
            The version of the instance of this dataset.
        tags : list of str, optional
            The tags associated with the given instance of this dataset.
        ext : str, optional
            The file extension to use. If not given, the default extension is
            used.

        Returns
        -------
        str
            The appropariate filepath.
        """
        if self.singleton:
            return dataset_filepath(
                filename=self.fname(version=version, tags=tags, ext=ext),
                task=self.task,
                **self.kwargs,
            )
        return dataset_filepath(
            filename=self.fname(version=version, tags=tags, ext=ext),
            dataset_name=self.name,
            task=self.task,
            **self.kwargs,
        )