def perimeter(self):
        '''
        Sum of the length of all sides, float.
        '''
        return sum([a.distance(b) for a, b in self.pairs()])