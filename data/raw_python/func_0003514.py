def whiteness_index(self):
        """Index of "Whiteness" based on visible bands.
        Parameters
        ----------

        Output
        ------
        ndarray:
            whiteness index
        """
        mean_vis = (self.blue + self.green + self.red) / 3

        blue_absdiff = np.absolute(self._divide_zero(self.blue - mean_vis, mean_vis))
        green_absdiff = np.absolute(self._divide_zero(self.green - mean_vis, mean_vis))
        red_absdiff = np.absolute(self._divide_zero(self.red - mean_vis, mean_vis))

        return blue_absdiff + green_absdiff + red_absdiff