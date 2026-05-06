def copy_dataset_files(self, ds, incver=False, cb=None, **kwargs):
        """
        Copy only files and configs into the database.
        :param ds: The source dataset to copy
        :param cb: A progress callback, taking two parameters: cb(message, num_records)
        :return:
        """
        from ambry.orm import File

        tables = [File]

        return self._copy_dataset_copy(ds, tables, incver, cb, **kwargs)