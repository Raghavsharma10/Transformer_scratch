def flip(self):
        '''
        :returns: None

        Swaps the positions of A and B.
        '''
        tmp = self.A.xyz
        self.A = self.B
        self.B = tmp