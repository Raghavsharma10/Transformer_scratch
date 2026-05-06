def load(self, yr=None, doy=None, date=None, fname=None, fid=None, 
             verifyPad=False):
        """Load instrument data into Instrument object .data.

        Parameters
        ----------
        yr : integer
            year for desired data
        doy : integer
            day of year
        date : datetime object
            date to load
        fname : 'string'
            filename to be loaded
        verifyPad : boolean 
            if True, padding data not removed (debug purposes)

        Returns
        --------
        Void.  Data is added to self.data

        Note
        ----
        Loads data for a chosen instrument into .data. Any functions chosen
        by the user and added to the custom processing queue (.custom.add)
        are automatically applied to the data before it is available to 
        user in .data.
        
        """
        # set options used by loading routine based upon user input
        if date is not None:
            self._set_load_parameters(date=date, fid=None)
            # increment 
            inc = pds.DateOffset(days=1)
            curr = date
        elif (yr is not None) & (doy is not None):
            date = pds.datetime(yr, 1, 1) + pds.DateOffset(days=(doy-1))
            self._set_load_parameters(date=date, fid=None)
            # increment 
            inc = pds.DateOffset(days=1)
            curr = self.date
        elif fname is not None:
            # date will have to be set later by looking at the data
            self._set_load_parameters(date=None,
                                      fid=self.files.get_index(fname))
            # increment one file at a time
            inc = 1
            curr = self._fid.copy()
        elif fid is not None:
            self._set_load_parameters(date=None, fid=fid)
            # increment one file at a time
            inc = 1
            curr = fid
        else:
            estr = 'Must supply a yr,doy pair, or datetime object, or filename'
            estr = '{:s} to load data from.'.format(estr)
            raise TypeError(estr)

        self.orbits._reset()
        # if pad  or multi_file_day is true, need to have a three day/file load
        loop_pad = self.pad if self.pad is not None else pds.DateOffset(seconds=0)   
        if (self.pad is not None) | self.multi_file_day:
            if self._next_data.empty & self._prev_data.empty:
                # data has not already been loaded for previous and next days
                # load data for all three
                print('Initializing three day/file window')
                # using current date or fid
                self._prev_data, self._prev_meta = self._load_prev()
                self._curr_data, self._curr_meta = \
                    self._load_data(date=self.date, fid=self._fid)
                self._next_data, self._next_meta = self._load_next()
            else:
                # moving forward in time
                if self._next_data_track == curr:
                    del self._prev_data
                    self._prev_data = self._curr_data
                    self._prev_meta = self._curr_meta
                    self._curr_data = self._next_data
                    self._curr_meta = self._next_meta
                    self._next_data, self._next_meta = self._load_next()
                # moving backward in time
                elif self._prev_data_track == curr:
                    del self._next_data
                    self._next_data = self._curr_data
                    self._next_meta = self._curr_meta
                    self._curr_data = self._prev_data
                    self._curr_meta = self._prev_meta
                    self._prev_data, self._prev_meta = self._load_prev()
                # jumped in time/or switched from filebased to date based access
                else:
                    del self._prev_data
                    del self._curr_data
                    del self._next_data
                    self._prev_data, self._prev_meta = self._load_prev()
                    self._curr_data, self._curr_meta = \
                                self._load_data(date=self.date, fid=self._fid)
                    self._next_data, self._next_meta = self._load_next()

            # make sure datetime indices for all data is monotonic
            if not self._prev_data.index.is_monotonic_increasing:
                self._prev_data.sort_index(inplace=True)
            if not self._curr_data.index.is_monotonic_increasing:
                self._curr_data.sort_index(inplace=True)
            if not self._next_data.index.is_monotonic_increasing:
                self._next_data.sort_index(inplace=True)
                
            # make tracking indexes consistent with new loads
            self._next_data_track = curr + inc
            self._prev_data_track = curr - inc
            # attach data to object
            if not self._curr_data.empty:
                self.data = self._curr_data.copy()
                self.meta = self._curr_meta.copy()
            else:
                self.data = DataFrame(None)
                # line below removed as it would delete previous meta, if any
                # if you end a seasonal analysis with a day with no data, then
                # no meta: self.meta = _meta.Meta()
            
            # multi file days can extend past a single day, only want data from 
            # specific date if loading by day
            # set up times for the possible data padding coming up
            if self._load_by_date:
                #print ('double trouble')
                first_time = self.date 
                first_pad = self.date - loop_pad
                last_time = self.date + pds.DateOffset(days=1) 
                last_pad = self.date + pds.DateOffset(days=1) + loop_pad
                want_last_pad = False
            # loading by file, can't be a multi_file-day flag situation
            elif (not self._load_by_date) and (not self.multi_file_day):
                #print ('single trouble')
                first_time = self._curr_data.index[0]
                first_pad = first_time - loop_pad
                last_time = self._curr_data.index[-1]
                last_pad = last_time + loop_pad
                want_last_pad = True
            else:
                raise ValueError("multi_file_day and loading by date are " + 
                                 "effectively equivalent.  Can't have " +
                                 "multi_file_day and load by file.")
            #print (first_pad, first_time, last_time, last_pad)

            # pad data based upon passed parameter
            if (not self._prev_data.empty) & (not self.data.empty):
                padLeft = self._prev_data.loc[first_pad : self.data.index[0]]
                if len(padLeft) > 0:
                    if (padLeft.index[-1] == self.data.index[0]) :
                        padLeft = padLeft.iloc[:-1, :]
                    self.data = pds.concat([padLeft, self.data])

            if (not self._next_data.empty) & (not self.data.empty):
                padRight = self._next_data.loc[self.data.index[-1] : last_pad]
                if len(padRight) > 0:
                    if (padRight.index[0] == self.data.index[-1]) :
                        padRight = padRight.iloc[1:, :]
                    self.data = pds.concat([self.data, padRight])
                    
            self.data = self.data.ix[first_pad : last_pad]
            # want exclusive end slicing behavior from above
            if not self.empty:
                if (self.data.index[-1] == last_pad) & (not want_last_pad):
                    self.data = self.data.iloc[:-1, :]
   
            ## drop any possible duplicate index times
            ##self.data.drop_duplicates(inplace=True)
            #self.data = self.data[~self.data.index.duplicated()]
            
        # if self.pad is False, load single day
        else:
            self.data, meta = self._load_data(date=self.date, fid=self._fid) 
            if not self.data.empty:
                self.meta = meta   
               
        # check if load routine actually returns meta
        if self.meta.data.empty:
            self.meta[self.data.columns] = {self.name_label: self.data.columns,
                                            self.units_label: [''] *
                                            len(self.data.columns)}
        # if loading by file set the yr, doy, and date
        if not self._load_by_date:
            if self.pad is not None:
                temp = first_time
            else:
                temp = self.data.index[0]
            self.date = pds.datetime(temp.year, temp.month, temp.day)
            self.yr, self.doy = utils.getyrdoy(self.date)

        if not self.data.empty:
            self._default_rtn(self)
        # clean
        if (not self.data.empty) & (self.clean_level != 'none'):
            self._clean_rtn(self)   
        # apply custom functions
        if not self.data.empty:
            self.custom._apply_all(self)
            
        # remove the excess padding, if any applied
        if (self.pad is not None) & (not self.data.empty) & (not verifyPad):
            self.data = self.data[first_time: last_time]
            if not self.empty:
                if (self.data.index[-1] == last_time) & (not want_last_pad):
                    self.data = self.data.iloc[:-1, :]

        # transfer any extra attributes in meta to the Instrument object
        self.meta.transfer_attributes_to_instrument(self)
        sys.stdout.flush()
        return