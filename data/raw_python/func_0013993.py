def get_new(self):
        """List new files since last recorded file state.
        
        pysat stores filenames in the user_home/.pysat directory. Returns
        a list of all new fileanmes since the last known change to files.
        Filenames are stored if there is a change and either update_files
        is True at instrument object level or files.refresh() is called.
        
        Returns
        -------
        pandas.Series
            files are indexed by datetime

        """

        # refresh files
        self.refresh()
        # current files
        new_info = self._load()
        # previous set of files
        old_info = self._load(prev_version=True)
        new_files = new_info[-new_info.isin(old_info)]
        return new_files