def dump_df(self, df, version=None, tags=None, ext=None, **kwargs):
        """Dumps an instance of this dataset into a file.

        Parameters
        ----------
        df : pandas.DataFrame
            The dataframe to dump to file.
        version: str, optional
            The version of the instance of this dataset.
        tags : list of str, optional
            The tags associated with the given instance of this dataset.
        ext : str, optional
            The file extension to use. If not given, the default extension is
            used.
        **kwargs : extra keyword arguments, optional
            Extra keyword arguments are forwarded to the serialization method
            of the SerializationFormat object corresponding to the extension
            used.
        """
        if ext is None:
            ext = self.default_ext
        fpath = self.fpath(version=version, tags=tags, ext=ext)
        fmt = SerializationFormat.by_name(ext)
        fmt.serialize(df, fpath, **kwargs)