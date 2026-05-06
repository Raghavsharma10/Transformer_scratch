def _count_repo_files(self):
        """Count the number of files in the data repositories.

        `_COUNT_FILE_TYPES` are used to determine which file types are checked
        explicitly.
        `_IGNORE_FILES` determine which files are ignored in (most) counts.

        Returns
        -------
        repo_files : int
            Total number of (non-ignored) files in all data repositories.

        """
        self.log.warning("Files:")
        num_files = 0
        repos = self.catalog.PATHS.get_all_repo_folders()
        num_type = np.zeros(len(self._COUNT_FILE_TYPES), dtype=int)
        num_ign = 0
        for rep in repos:
            # Get the last portion of the filepath for this repo
            last_path = _get_last_dirs(rep, 2)
            # Get counts for different file types
            n_all = self._count_files_by_type(rep, '*')
            n_type = np.zeros(len(self._COUNT_FILE_TYPES), dtype=int)
            for ii, ftype in enumerate(self._COUNT_FILE_TYPES):
                n_type[ii] = self._count_files_by_type(rep, '*.' + ftype)
            # Get the number of ignored files
            # (total including ignore, minus 'all')
            n_ign = self._count_files_by_type(rep, '*', ignore=False)
            n_ign -= n_all
            f_str = self._file_nums_str(n_all, n_type, n_ign)
            f_str = "{}: {}".format(last_path, f_str)
            self.log.warning(f_str)
            # Update cumulative counts
            num_files += n_all
            num_type += n_type
            num_ign += n_ign

        f_str = self._file_nums_str(num_files, num_type, num_ign)
        self.log.warning(f_str)
        return num_files