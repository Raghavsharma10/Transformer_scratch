def update_tile_extent_bounds(self):
        """
        Updates the :attr:`tile_beg_min` and :attr:`tile_end_max`
        data members according to :attr:`tile_bounds_policy`.
        """

        if self.tile_bounds_policy == NO_BOUNDS:
            self.tile_beg_min = self.array_start - self.halo[:, 0]
            self.tile_end_max = self.array_start + self.array_shape + self.halo[:, 1]
        elif self.tile_bounds_policy == ARRAY_BOUNDS:
            self.tile_beg_min = self.array_start
            self.tile_end_max = self.array_start + self.array_shape