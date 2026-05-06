def download(self, version=None, tags=None, ext=None, overwrite=False,
                 verbose=False, **kwargs):
        """Downloads the given instance of this dataset from dataset store.

        Parameters
        ----------
        version: str, optional
            The version of the instance of this dataset.
        tags : list of str, optional
            The tags associated with the given instance of this dataset.
        ext : str, optional
            The file extension to use. If not given, the default extension is
            used.
        overwrite : bool, default False
            If set to True, the given instance of the dataset is downloaded
            from dataset store even if it exists in the local data directory.
            Otherwise, if a matching dataset is found localy, download is
            skipped.
        verbose : bool, default False
            If set to True, informative messages are printed.
        **kwargs : extra keyword arguments
            Extra keyword arguments are forwarded to
            azure.storage.blob.BlockBlobService.get_blob_to_path.
        """
        fpath = self.fpath(version=version, tags=tags, ext=ext)
        if os.path.isfile(fpath) and not overwrite:
            if verbose:
                print(
                    "File exists and overwrite set to False, so not "
                    "downloading {} with version={} and tags={}".format(
                        self.name, version, tags))
                return
        download_dataset(
            dataset_name=self.name,
            file_path=fpath,
            task=self.task,
            dataset_attributes=self.kwargs,
            **kwargs,
        )