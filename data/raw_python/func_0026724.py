def _count_files_by_type(self, path, pattern, ignore=True):
        """Count files in the given path, with the given pattern.

        If `ignore = True`, skip files in the `_IGNORE_FILES` list.

        Returns
        -------
        num_files : int

        """
        # Get all files matching the given path and pattern
        files = glob(os.path.join(path, pattern))
        # Count the files
        files = [ff for ff in files
                 if os.path.split(ff)[-1] not in self._IGNORE_FILES
                 or not ignore]
        num_files = len(files)
        return num_files