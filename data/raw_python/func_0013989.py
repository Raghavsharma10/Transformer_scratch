def _attach_files(self, files_info):
        """Attaches info returned by instrument list_files routine to
        Instrument object.
        """

        if not files_info.empty:
            if (len(files_info.index.unique()) != len(files_info)):
                estr = 'WARNING! Duplicate datetimes in provided file '
                estr = '{:s}information.\nKeeping one of each '.format(estr)
                estr = '{:s}of the duplicates, dropping the rest.'.format(estr)
                print(estr)
                print(files_info.index.get_duplicates())

                idx = np.unique(files_info.index, return_index=True)
                files_info = files_info.ix[idx[1]]
                #raise ValueError('List of files must have unique datetimes.')

            self.files = files_info.sort_index()
            date = files_info.index[0]
            self.start_date = pds.datetime(date.year, date.month, date.day)
            date = files_info.index[-1]
            self.stop_date = pds.datetime(date.year, date.month, date.day)
        else:
            self.start_date = None
            self.stop_date = None
            # convert to object type
            # necessary if Series is empty, enables == checks with strings
            self.files = files_info.astype(np.dtype('O'))