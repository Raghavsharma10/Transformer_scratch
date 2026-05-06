def upload(self, version=None, tags=None, ext=None, source_fpath=None,
               overwrite=False, **kwargs):
        """Uploads the given instance of this dataset to dataset store.

        Parameters
        ----------
        version: str, optional
            The version of the instance of this dataset.
        tags : list of str, optional
            The tags associated with the given instance of this dataset.
        ext : str, optional
            The file extension to use. If not given, the default extension is
            used. If source_fpath is given, this is ignored, and the extension
            of the source f
        source_fpath : str, optional
            The full path for the source file to use. If given, the file is
            copied from the given path to the local storage path before
            uploading.
        **kwargs : extra keyword arguments
            Extra keyword arguments are forwarded to
            azure.storage.blob.BlockBlobService.create_blob_from_path.
        """
        if source_fpath:
            ext = self.add_local(
                source_fpath=source_fpath, version=version, tags=tags)
        if ext is None:
            ext = self._find_extension(version=version, tags=tags)
        if ext is None:
            attribs = "{}{}".format(
                "version={} and ".format(version) if version else "",
                "tags={}".format(tags) if tags else "",
            )
            raise MissingDatasetError(
                "No dataset with {} in local store!".format(attribs))
        fpath = self.fpath(version=version, tags=tags, ext=ext)
        if not os.path.isfile(fpath):
            attribs = "{}{}ext={}".format(
                "version={} and ".format(version) if version else "",
                "tags={} and ".format(tags) if tags else "",
                ext,
            )
            raise MissingDatasetError(
                "No dataset with {} in local store! (path={})".format(
                    attribs, fpath))
        upload_dataset(
            dataset_name=self.name,
            file_path=fpath,
            task=self.task,
            dataset_attributes=self.kwargs,
            **kwargs,
        )