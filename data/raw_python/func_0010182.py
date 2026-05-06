def cross(self, other):
        '''
        :other: Point or point equivalent
        :return: float

        Vector cross product of points U (self) and V (other), computed:

        U x V = (u1*i + u2*j + u3*k) x (v1*i + v2*j + v3*k)
        s1 = u2v3 - u3v2
        s2 = u3v1 - u1v3
        s3 = u1v2 - u2v1

        U x V = s1 + s2 + s3

        Returns a float.
        '''

        b = self.__class__._convert(other)

        return sum([(self.y * b.z) - (self.z * b.y),
                    (self.z * b.x) - (self.x * b.z),
                    (self.x * b.y) - (self.y * b.x)])