def check_tile_bounds_policy(self):
        """
        Raises :obj:`ValueError` if :attr:`tile_bounds_policy`
        is not in :samp:`[{self}.ARRAY_BOUNDS, {self}.NO_BOUNDS]`.
        """
        if self.tile_bounds_policy not in self.valid_tile_bounds_policies:
            raise ValueError(
                "Got self.tile_bounds_policy=%s, which is not in %s."
                %
                (self.tile_bounds_policy, self.valid_tile_bounds_policies)
            )