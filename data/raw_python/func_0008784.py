def do_march_all(self):
        """
        Recursive march in the case that we have a fragmented shape.

        Returns
        -------
        perimeters : [perimeter1, ...]
           The perimeters of all the regions in the image.

        See Also
        --------
        :func:`AegeanTools.msq2.MarchingSquares.do_march`
        """
        # copy the data since we are going to be modifying it
        data_copy = copy(self.data)

        # iterate through finding an island, creating a perimeter,
        # and then blanking the island
        perimeters = []
        p = self.find_start_point()
        while p is not None:
            x, y = p
            perim = self.walk_perimeter(x, y)
            perimeters.append(perim)
            self._blank_within(perim)
            p = self.find_start_point()

        # restore the data
        self.data = data_copy
        return perimeters