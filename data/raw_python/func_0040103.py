def upload_df(self, df, version=None, tags=None, ext=None, **kwargs):
        """Dumps an instance of this dataset into a file and then uploads it
        to dataset store.

        Parameters
        ----------
        df : pandas.DataFrame
            The dataframe to dump and upload.
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
        self.dump_df(df=df, version=version, tags=tags, ext=ext, **kwargs)
        self.upload(version=version, tags=tags, ext=ext)