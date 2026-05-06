def intersection(self, other):
        """Returns a MOC representing the intersection with another MOC.

        >>> p = MOC(2, (3, 4, 5))
        >>> q = MOC(2, (4, 5, 6))
        >>> p.intersection(q)
        <MOC: [(2, [4, 5])]>
        """

        inter = MOC()

        for (order, cells) in other:
            for cell in cells:
                for i in self._compare_operation(order, cell, True, 'inter'):
                    inter.add(*i)

        return inter