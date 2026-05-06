def put_item(self, fpath, relpath):
        """Put item with content from fpath at relpath in dataset.

        Missing directories in relpath are created on the fly.

        :param fpath: path to the item on disk
        :param relpath: relative path name given to the item in the dataset as
                        a handle
        :returns: the handle given to the item
        """

        # Define the destination path and make any missing parent directories.
        dest_path = os.path.join(self._data_abspath, relpath)
        dirname = os.path.dirname(dest_path)
        mkdir_parents(dirname)

        # Copy the file across.
        shutil.copyfile(fpath, dest_path)

        return relpath