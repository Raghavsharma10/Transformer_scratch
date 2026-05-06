def reverse_reference(self):
        '''Changes the coordinates as if the reference sequence has been reverse complemented'''
        self.ref_start = self.ref_length - self.ref_start - 1
        self.ref_end = self.ref_length - self.ref_end - 1