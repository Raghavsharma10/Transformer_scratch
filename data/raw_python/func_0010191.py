def altitudes(self):
        '''
        A list of the altitudes of each vertex [AltA, AltB, AltC], list of
        floats.

        An altitude is the shortest distance from a vertex to the side
        opposite of it.

        '''
        A = self.area * 2

        return [A / self.a, A / self.b, A / self.c]