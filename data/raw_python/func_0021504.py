def lerp(self, a, t):
        """ Lerp. Linear interpolation from self to a"""
        return self.plus(a.minus(self).times(t));