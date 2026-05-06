def set_size(self, position, size, padding):
        """Row size setter.

        The size correspond to the row height, since the row width is constraint
        to the surface width the associated keyboard belongs. Once size is settled,
        the size for each child keys is associated.
        
        :param position: Position of this row.
        :param size: Size of the row (height)
        :param padding: Padding between key.
        """
        self.height = size
        self.position = position
        x = position[0]
        for key in self.keys:
            key.set_size(size)
            key.position = (x, position[1])
            x += padding + key.size[0]