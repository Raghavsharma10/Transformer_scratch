def _swap(self):
        '''Swaps the alignment so that the reference becomes the query and vice-versa. Swaps their names, coordinates etc. The frame is not changed'''
        self.ref_start, self.qry_start = self.qry_start, self.ref_start
        self.ref_end, self.qry_end = self.qry_end, self.ref_end
        self.hit_length_ref, self.hit_length_qry = self.hit_length_qry, self.hit_length_ref
        self.ref_length, self.qry_length = self.qry_length, self.ref_length
        self.ref_name, self.qry_name = self.qry_name, self.ref_name