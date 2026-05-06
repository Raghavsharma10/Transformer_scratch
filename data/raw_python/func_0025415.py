def size_to_content(self):
        """ Size the canvas item to the proper width. """
        new_sizing = self.copy_sizing()
        new_sizing.minimum_width = 0
        new_sizing.maximum_width = 0
        axes = self.__axes
        if axes and axes.is_valid:
            if axes.y_calibration and axes.y_calibration.units:
                new_sizing.minimum_width = self.font_size + 4
                new_sizing.maximum_width = self.font_size + 4
        self.update_sizing(new_sizing)