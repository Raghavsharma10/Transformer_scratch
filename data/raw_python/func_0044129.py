def copy(self):
        """Return a copy of a MOC.

        >>> p = MOC(4, (5, 6))
        >>> q = p.copy()
        >>> repr(q)
        '<MOC: [(4, [5, 6])]>'
        """

        copy = MOC(name=self.name, mocid=self.id,
                   origin=self.origin, moctype=self.type)

        copy += self

        return copy