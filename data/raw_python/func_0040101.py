def df(self, version=None, tags=None, ext=None, **kwargs):
        """Loads an instance of this dataset into a dataframe.

        Parameters
        ----------
        version: str, optional
            The version of the instance of this dataset.
        tags : list of str, optional
            The tags associated with the desired instance of this dataset.
        ext : str, optional
            The file extension to use. If not given, the default extension is
            used.
        **kwargs : extra keyword arguments, optional
            Extra keyword arguments are forwarded to the deserialization method
            of the SerializationFormat object corresponding to the extension
            used.

        Returns
        -------
        pandas.DataFrame
            A dataframe containing the desired instance of this dataset.
        """
        ext = self._find_extension(version=version, tags=tags)
        if ext is None:
            attribs = "{}{}".format(
                "version={} and ".format(version) if version else "",
                "tags={}".format(tags) if tags else "",
            )
            raise MissingDatasetError(
                "No dataset with {} in local store!".format(attribs))
        fpath = self.fpath(version=version, tags=tags, ext=ext)
        fmt = SerializationFormat.by_name(ext)
        return fmt.deserialize(fpath, **kwargs)