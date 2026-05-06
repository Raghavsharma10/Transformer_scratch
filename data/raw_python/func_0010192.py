def isIsosceles(self):
        '''
        True iff two side lengths are equal, boolean.
        '''
        return (self.a == self.b) or (self.a == self.c) or (self.b == self.c)