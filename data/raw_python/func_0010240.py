def isEquilateral(self):
        '''
        True if all sides of the triangle are the same length.

        All equilateral triangles are also isosceles.
        All equilateral triangles are also acute.

        '''
        if not nearly_eq(self.a, self.b):
            return False

        if not nearly_eq(self.b, self.c):
            return False

        return nearly_eq(self.a, self.c)