def ref_coords(self):
        '''Returns a pyfastaq.intervals.Interval object of the start and end coordinates in the reference sequence'''
        return pyfastaq.intervals.Interval(min(self.ref_start, self.ref_end), max(self.ref_start, self.ref_end))