def isosceles(cls, origin=None, base=1, alpha=90):
        '''
        :origin: optional Point
        :base: optional float describing triangle base length
        :return: Triangle initialized with points comprising a
                 isosceles triangle.

        XXX isoceles triangle definition

        '''
        o = Point(origin)
        base = o.x + base

        return cls(o, [base, o.y], [base / 2, o.y + base])