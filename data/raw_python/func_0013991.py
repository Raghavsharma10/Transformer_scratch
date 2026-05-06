def _load(self, prev_version=False):
        """Load stored filelist and return as Pandas Series

        Parameters
        ----------
        prev_version : boolean
            if True, will load previous version of file list

        Returns
        -------
        pandas.Series
            Full path file names are indexed by datetime
            Series is empty if there is no file list to load
        """

        fname = self.stored_file_name
        if prev_version:
            fname = os.path.join(self.home_path, 'previous_'+fname)
        else:
            fname = os.path.join(self.home_path, fname)

        if os.path.isfile(fname) and (os.path.getsize(fname) > 0):
            if self.write_to_disk:
                return pds.read_csv(fname, index_col=0, parse_dates=True,
                                    squeeze=True, header=None)
            else:
                # grab files from memory
                if prev_version:
                    return self._previous_file_list
                else:
                    return self._current_file_list
        else:
            return pds.Series([], dtype='a')