def reverse_query(self):
        '''Changes the coordinates as if the query sequence has been reverse complemented'''
        self.qry_start = self.qry_length - self.qry_start - 1
        self.qry_end = self.qry_length - self.qry_end - 1