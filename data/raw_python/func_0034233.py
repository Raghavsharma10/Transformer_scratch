def configure_bound(self, surface_size):
        """Compute keyboard bound regarding of this layout.
        
        If key_size is None, then it will compute it regarding of the given surface_size.

        :param surface_size: Size of the surface this layout will be rendered on.
        :raise ValueError: If the layout model is empty.
        """
        r = len(self.rows)
        max_length = self.max_length
        if self.key_size is None:
            self.key_size = (surface_size[0] - (self.padding * (max_length + 1))) / max_length
        height = self.key_size * r + self.padding * (r + 1)
        if height >= surface_size[1] / 2:
            logger.warning('Computed keyboard height outbound target surface, reducing key_size to match')
            self.key_size = ((surface_size[1] / 2) - (self.padding * (r + 1))) / r
            height = self.key_size * r + self.padding * (r + 1)
            logger.warning('Normalized key_size to %spx' % self.key_size)
        self.set_size((surface_size[0], height), surface_size)