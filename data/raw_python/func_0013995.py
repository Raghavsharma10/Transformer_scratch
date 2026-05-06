def get_file_array(self, start, end):
        """Return a list of filenames between and including start and end.
        
        Parameters
        ----------
            start: array_like or single string 
                filenames for start of returned filelist
            stop: array_like or single string
                filenames inclusive end of list
                
        Returns
        -------
            list of filenames between and including start and end over all
            intervals. 
            
        """
        if hasattr(start, '__iter__') & hasattr(end, '__iter__'):
            files = []
            for (sta,stp) in zip(start, end):
                id1 = self.get_index(sta)
                id2 = self.get_index(stp)
                files.extend(self.files.iloc[id1 : id2+1])
        elif hasattr(start, '__iter__') | hasattr(end, '__iter__'):
            estr = 'Either both or none of the inputs need to be iterable'
            raise ValueError(estr)
        else:
            id1 = self.get_index(start)
            id2 = self.get_index(end)
            files = self.files[id1:id2+1].to_list()
        return files