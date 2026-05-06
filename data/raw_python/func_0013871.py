def _load_next(self):
        """Load the next days data (or file) without incrementing the date.
        Repeated calls will not advance date/file and will produce the same data
        
        Uses info stored in object to either increment the date, 
        or the file. Looks for self._load_by_date flag. 
         
        """
        if self._load_by_date:
            next_date = self.date + pds.DateOffset(days=1)
            return self._load_data(date=next_date)
        else:
            return self._load_data(fid=self._fid+1)