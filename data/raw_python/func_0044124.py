def area(self):
        """The area enclosed by the MOC, in steradians.

        >>> m = MOC(0, (0, 1, 2))
        >>> round(m.area, 2)
        3.14
        """

        self.normalize()
        area = 0.0

        for (order, cells) in self:
            area += (len(cells) * pi) / (3 * 4 ** order)

        return area