def next(self, verifyPad=False):
        """Manually iterate through the data loaded in Instrument object.
        
        Bounds of iteration and iteration type (day/file) are set by 
        `bounds` attribute.
        
        Note
        ----
        If there were no previous calls to load then the 
        first day(default)/file will be loaded.
         
        """
        
        if self._iter_type == 'date':
            if self.date is not None:
                idx, = np.where(self._iter_list == self.date)
                if (len(idx) == 0):
                    raise StopIteration('File list is empty. Nothing to be done.')
                elif idx[-1]+1 >= len(self._iter_list):
                    raise StopIteration('Outside the set date boundaries.')
                else:
                    idx += 1
                    self.load(date=self._iter_list[idx[0]], verifyPad=verifyPad)
            else:
                self.load(date=self._iter_list[0], verifyPad=verifyPad)

        elif self._iter_type == 'file':
            if self._fid is not None:
                first = self.files.get_index(self._iter_list[0])
                last = self.files.get_index(self._iter_list[-1])
                if (self._fid < first) | (self._fid+1 > last):
                    raise StopIteration('Outside the set file boundaries.')
                else:
                    self.load(fname=self._iter_list[self._fid+1-first],
                              verifyPad=verifyPad)
            else:
                self.load(fname=self._iter_list[0], verifyPad=verifyPad)