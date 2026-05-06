def archive(self,prepend=None,overwrite=no,quiet=yes):
        """ Create backup copies of the WCS keywords with the given prepended
            string.
            If backup keywords are already present, only update them if
            'overwrite' is set to 'yes', otherwise, do warn the user and do nothing.
            Set the WCSDATE at this time as well.
        """
        # Verify that existing backup values are not overwritten accidentally.
        if len(list(self.backup.keys())) > 0 and overwrite == no:
            if not quiet:
                print('WARNING: Backup WCS keywords already exist! No backup made.')
                print('         The values can only be overridden if overwrite=yes.')
            return

        # Establish what prepend string to use...
        if prepend is None:
            if self.prepend is not None:
                _prefix = self.prepend
            else:
                _prefix = DEFAULT_PREFIX
        else:
            _prefix = prepend

        # Update backup and orig_wcs dictionaries
        # We have archive keywords and a defined prefix
        # Go through and append them to self.backup
        self.prepend = _prefix
        for key in self.wcstrans.keys():
            if key != 'pixel scale':
                _archive_key = self._buildNewKeyname(key,_prefix)
            else:
                _archive_key = self.prepend.lower()+'pscale'
#            if key != 'pixel scale':
            self.orig_wcs[_archive_key] = self.__dict__[self.wcstrans[key]]
            self.backup[key] = _archive_key
            self.revert[_archive_key] = key

        # Setup keyword to record when these keywords were backed up.
        self.orig_wcs['WCSCDATE']= fileutil.getLTime()
        self.backup['WCSCDATE'] = 'WCSCDATE'
        self.revert['WCSCDATE'] = 'WCSCDATE'