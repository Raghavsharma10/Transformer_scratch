def cells(self):
        """The number of cells in the MOC.

        This gives the total number of cells at all orders,
        with cells from every order counted equally.

        >>> m = MOC(0, (1, 2))
        >>> m.cells
        2
        """

        n = 0

        for (order, cells) in self:
            n += len(cells)

        return n