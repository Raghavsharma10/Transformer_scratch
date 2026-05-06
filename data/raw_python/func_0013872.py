def _load_prev(self):
        """Load the next days data (or file) without decrementing the date.
        Repeated calls will not decrement date/file and will produce the same
        data
        
        Uses info stored in object to either decrement the date, 
        or the file. Looks for self._load_by_date flag.  
        
        """

        if self._load_by_date:
            prev_date = self.date - pds.DateOffset(days=1)
            return self._load_data(date=prev_date)
        else:
            return self._load_data(fid=self._fid-1)