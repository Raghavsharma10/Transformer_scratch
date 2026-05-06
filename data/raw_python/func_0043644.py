def set_split_extents_by_tile_max_bytes(self):
        """
        Sets split extents (:attr:`split_begs`
        and :attr:`split_ends`) calculated using
        from :attr:`max_tile_bytes`
        (and :attr:`max_tile_shape`, :attr:`sub_tile_shape`, :attr:`halo`).

        """
        self.tile_shape = \
            calculate_tile_shape_for_max_bytes(
                array_shape=self.array_shape,
                array_itemsize=self.array_itemsize,
                max_tile_bytes=self.max_tile_bytes,
                max_tile_shape=self.max_tile_shape,
                sub_tile_shape=self.sub_tile_shape,
                halo=self.halo
            )
        self.set_split_extents_by_tile_shape()