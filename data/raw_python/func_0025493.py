def color_map_data(self) -> typing.Optional[numpy.ndarray]:
        """Return the color map data as a uint8 ndarray with shape (256, 3)."""
        if self.display_data_shape is None:  # is there display data?
            return None
        else:
            return self.__color_map_data if self.__color_map_data is not None else ColorMaps.get_color_map_data_by_id("grayscale")