def is_self_hit(self):
        '''Returns true iff the alignment is of a sequence to itself: names and all coordinates are the same and 100 percent identity'''
        return self.ref_name == self.qry_name \
                and self.ref_start == self.qry_start \
                and self.ref_end == self.qry_end \
                and self.percent_identity == 100