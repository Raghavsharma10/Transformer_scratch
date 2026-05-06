def _prep_git_add_file_list(self,
                                repo,
                                size_limit,
                                fail=True,
                                file_types=None):
        """Get a list of files which should be added to the given repository.

        Notes
        -----
        * Finds files in the *root* of the given repository path.
        * If `file_types` is given, only use those file types.
        * If an uncompressed file is above the `size_limit`, it is compressed.
        * If a compressed file is above the file limit, an error is raised
          (if `fail = True`) or it is skipped (if `fail == False`).

        Arguments
        ---------
        repo : str
            Path to repository
        size_limit : scalar
        fail : bool
            Raise an error if a compressed file is still above the size limit.
        file_types : list of str or None
            Exclusive list of file types to add. 'None' to add all filetypes.

        """
        add_files = []
        if file_types is None:
            file_patterns = ['*']
        else:
            self.log.error(
                "WARNING: uncertain behavior with specified file types!")
            file_patterns = ['*.' + ft for ft in file_types]

        # Construct glob patterns for each file-type
        file_patterns = [os.path.join(repo, fp) for fp in file_patterns]
        for pattern in file_patterns:
            file_list = glob(pattern)
            for ff in file_list:
                fsize = os.path.getsize(ff)
                fname = str(ff)
                comp_failed = False
                # If the found file is too large
                if fsize > size_limit:
                    self.log.debug("File '{}' size '{}' MB.".format(
                        fname, fsize / 1028 / 1028))
                    # If the file is already compressed... fail or skip
                    if ff.endswith('.gz'):
                        self.log.error(
                            "File '{}' is already compressed.".format(fname))
                        comp_failed = True
                    # Not yet compressed - compress it
                    else:
                        fname = compress_gz(fname)
                        fsize = os.path.getsize(fname)
                        self.log.info("Compressed to '{}', size '{}' MB".
                                      format(fname, fsize / 1028 / 1028))
                        # If still too big, fail or skip
                        if fsize > size_limit:
                            comp_failed = True

                # If compressed file is too large, skip file or raise error
                if comp_failed:
                    # Raise an error
                    if fail:
                        raise RuntimeError(
                            "File '{}' cannot be added!".format(fname))
                    # Skip file without adding it
                    self.log.info("Skipping file.")
                    continue

                # If everything is good, add file to list
                add_files.append(fname)

        return add_files