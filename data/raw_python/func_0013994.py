def get_index(self, fname):
        """Return index for a given filename. 
        
        Parameters
        ----------
        fname : string
            filename
        
        Note
        ----
        If fname not found in the file information already attached 
        to the instrument.files instance, then a files.refresh() call 
        is made.
        
        """

        idx, = np.where(fname == self.files)
        if len(idx) == 0:
            # filename not in index, try reloading files from disk
            self.refresh()
            #print("DEBUG get_index:", fname, self.files)
            idx, = np.where(fname == np.array(self.files))

            if len(idx) == 0:
                raise ValueError('Could not find "' + fname +
                                 '" in available file list. Valid Example: ' +
                                 self.files.iloc[0])
        # return a scalar rather than array - otherwise introduces array to
        # index warnings.
        return idx[0]