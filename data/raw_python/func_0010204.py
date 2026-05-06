def vertices(self):
        '''
        A dictionary of four points where the axes intersect the ellipse, dict.
        '''
        return {'a': self.a, 'a_neg': self.a_neg,
                'b': self.b, 'b_neg': self.b_neg}