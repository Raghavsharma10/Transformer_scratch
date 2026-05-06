def congruent(self, other):
        '''
        A congruent B

        True iff all angles of 'A' equal angles in 'B' and
        all side lengths of 'A' equal all side lengths of 'B', boolean.

        '''

        a = set(self.angles)
        b = set(other.angles)

        if len(a) != len(b) or len(a.difference(b)) != 0:
            return False

        a = set(self.sides)
        b = set(other.sides)

        return len(a) == len(b) and len(a.difference(b)) == 0