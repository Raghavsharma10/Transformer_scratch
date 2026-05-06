def altitudes(self):
        '''
        A list of the altitudes of each vertex [AltA, AltB, AltC], list of
        floats.

        An altitude is the shortest distance from a vertex to the side
        opposite of it.

        '''
        a = self.area * 2

        return [a / self.a, a / self.b, a / self.c]