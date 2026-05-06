def fname(self, version=None, tags=None, ext=None):
        """Returns the filename appropriate for an instance of this dataset.

        Parameters
        ----------
        version: str, optional
            The version of the instance of this dataset.
        tags : list of str, optional
            The tags associated with the instance of this dataset.
        ext : str, optional
            The file extension to use. If not given, the default extension is
            used.

        Returns
        -------
        str
            The appropariate filename.
        """
        if ext is None:
            ext = self.default_ext
        return '{}{}{}.{}'.format(
            self.fname_base,
            self._tags_to_str(tags=tags),
            self._version_to_str(version=version),
            ext,
        )