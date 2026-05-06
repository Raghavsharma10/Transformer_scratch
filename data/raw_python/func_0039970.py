def get(self):
        """
        Get an rgb color tuple according to the probability distribution.

        Returns:
            tuple(int, int, int): A ``(red, green, blue)`` tuple.

        Example:
            >>> color = SoftColor(([(0, 1), (255, 10)],),
            ...                   ([(0, 1), (255, 10)],),
            ...                   ([(0, 1), (255, 10)],))
            >>> color.get()                                    # doctest: +SKIP
            (234, 201, 243)
        """
        if isinstance(self.red, SoftInt):
            red = self.red.get()
        else:
            red = self.red
        if isinstance(self.green, SoftInt):
            green = self.green.get()
        else:
            green = self.green
        if isinstance(self.blue, SoftInt):
            blue = self.blue.get()
        else:
            blue = self.blue
        return (red, green, blue)