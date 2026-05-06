def qry_coords(self):
        '''Returns a pyfastaq.intervals.Interval object of the start and end coordinates in the query sequence'''
        return pyfastaq.intervals.Interval(min(self.qry_start, self.qry_end), max(self.qry_start, self.qry_end))