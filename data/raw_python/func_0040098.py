def add_local(self, source_fpath, version=None, tags=None):
        """Copies a given file into local store as an instance of this dataset.

        Parameters
        ----------
        source_fpath : str
            The full path for the source file to use.
        version: str, optional
            The version of the instance of this dataset.
        tags : list of str, optional
            The tags associated with the given instance of this dataset.

        Returns
        -------
        ext : str
            The extension of the file added.
        """
        ext = os.path.splitext(source_fpath)[1]
        ext = ext[1:]  # we dont need the dot
        fpath = self.fpath(version=version, tags=tags, ext=ext)
        shutil.copyfile(src=source_fpath, dst=fpath)
        return ext