def get_radius_indexes(self, radius, max_ranges=None):
        """Return the indexes of the interacting neighboring unit cells

           Interacting neighboring unit cells have at least one point in their
           box volume that has a distance smaller or equal than radius to at
           least one point in the central cell. This concept is of importance
           when computing pair wise long-range interactions in periodic systems.

           Argument:
            | ``radius``  --  the radius of the interaction sphere

           Optional argument:
            | ``max_ranges``  --  numpy array with three elements: The maximum
                                  ranges of indexes to consider. This is
                                  practical when working with the minimum image
                                  convention to reduce the generated bins to the
                                  minimum image. (see binning.py) Use -1 to
                                  avoid such limitations. The default is three
                                  times -1.

        """
        if max_ranges is None:
            max_ranges = np.array([-1, -1, -1])
        ranges = self.get_radius_ranges(radius)*2+1
        mask = (max_ranges != -1) & (max_ranges < ranges)
        ranges[mask] = max_ranges[mask]
        max_size = np.product(self.get_radius_ranges(radius)*2 + 1)
        indexes = np.zeros((max_size, 3), int)

        from molmod.ext import unit_cell_get_radius_indexes
        reciprocal = self.reciprocal*self.active
        matrix = self.matrix*self.active
        size = unit_cell_get_radius_indexes(
            matrix, reciprocal, radius, max_ranges, indexes
        )
        return indexes[:size]