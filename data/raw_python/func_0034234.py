def set_size(self, size, surface_size):
        """Sets the size of this layout, and updates
        position, and rows accordingly.

        :param size: Size of this layout.
        :param surface_size: Target surface size on which layout will be displayed.
        """
        self.size = size
        self.position = (0, surface_size[1] - self.size[1])
        y = self.position[1] + self.padding
        max_length = self.max_length
        for row in self.rows:
            r = len(row)
            width = (r * self.key_size) + ((r + 1) * self.padding)
            x = (surface_size[0] - width) / 2
            if row.space is not None:
                x -= ((row.space.length - 1) * self.key_size) / 2
            row.set_size((x, y), self.key_size, self.padding)
            y += self.padding + self.key_size