def color_map_data(self, data: numpy.ndarray) -> None:
        """Set the data and mark the canvas item for updating.

        Data should be an ndarray of shape (256, 3) with type uint8
        """
        self.__color_map_data = data
        self.update()