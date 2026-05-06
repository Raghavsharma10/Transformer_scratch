def ndsi(self):
        """ Normalized difference snow index.
        :return: NDSI
        """
        green, swir1 = self.reflectance(3), self.reflectance(6)
        ndsi = self._divide_zero((green - swir1), (green + swir1), nan)

        return ndsi