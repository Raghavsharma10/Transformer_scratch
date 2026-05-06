def circumradius(self):
        '''
        Distance from the circumcenter to all the verticies in
        the Triangle, float.

        '''
        return (self.a * self.b * self.c) / (self.area * 4)