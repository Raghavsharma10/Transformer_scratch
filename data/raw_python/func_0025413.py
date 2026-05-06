def size_to_content(self):
        """ Size the canvas item to the proper height. """
        new_sizing = self.copy_sizing()
        new_sizing.minimum_height = 0
        new_sizing.maximum_height = 0
        axes = self.__axes
        if axes and axes.is_valid:
            if axes.x_calibration and axes.x_calibration.units:
                new_sizing.minimum_height = self.font_size + 4
                new_sizing.maximum_height = self.font_size + 4
        self.update_sizing(new_sizing)