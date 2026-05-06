def download(self, start, stop, freq='D', user=None, password=None,
                 **kwargs):
        """Download data for given Instrument object from start to stop.
        
        Parameters
        ----------
        start : pandas.datetime
            start date to download data
        stop : pandas.datetime
            stop date to download data
        freq : string
            Stepsize between dates for season, 'D' for daily, 'M' monthly 
            (see pandas)
        user : string
            username, if required by instrument data archive
        password : string
            password, if required by instrument data archive
        **kwargs : dict
            Dictionary of keywords that may be options for specific instruments
            
        Note
        ----
        Data will be downloaded to pysat_data_dir/patform/name/tag
        
        If Instrument bounds are set to defaults they are updated
        after files are downloaded.
        
        """
        import errno
        # make sure directories are there, otherwise create them
        try:
            os.makedirs(self.files.data_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        print('Downloading data to: ', self.files.data_path)
        date_array = utils.season_date_range(start, stop, freq=freq)
        if user is None:
            self._download_rtn(date_array,
                               tag=self.tag,
                               sat_id=self.sat_id,
                               data_path=self.files.data_path,
                               **kwargs)
        else:
            self._download_rtn(date_array,
                               tag=self.tag,
                               sat_id=self.sat_id,
                               data_path=self.files.data_path,
                               user=user,
                               password=password, **kwargs)
        # get current file date range
        first_date = self.files.start_date
        last_date = self.files.stop_date
            
        print('Updating pysat file list')
        self.files.refresh()

        # if instrument object has default bounds, update them
        if len(self.bounds[0]) == 1:
            if(self.bounds[0][0] == first_date and
               self.bounds[1][0] == last_date):
                print('Updating instrument object bounds.')
                self.bounds = None