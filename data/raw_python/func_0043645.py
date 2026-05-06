def set_split_extents(self):
        """
        Sets split extents (:attr:`split_begs`
        and :attr:`split_ends`) calculated using
        selected attributes set from :meth:`__init__`.
        """

        self.check_split_parameters()
        self.update_tile_extent_bounds()

        if self.indices_per_axis is not None:
            self.set_split_extents_by_indices_per_axis()
        elif (self.split_size is not None) or (self.split_num_slices_per_axis is not None):
            self.set_split_extents_by_split_size()
        elif self.tile_shape is not None:
            self.set_split_extents_by_tile_shape()
        elif self.max_tile_bytes is not None:
            self.set_split_extents_by_tile_max_bytes()