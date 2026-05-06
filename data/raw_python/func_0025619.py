def on_same_strand(self):
        '''Returns true iff the direction of the alignment is the same in the reference and the query'''
        return (self.ref_start < self.ref_end) == (self.qry_start < self.qry_end)