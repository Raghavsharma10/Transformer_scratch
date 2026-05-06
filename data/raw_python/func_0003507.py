def ndvi(self):
        """ Normalized difference vegetation index.
        :return: NDVI
        """
        red, nir = self.reflectance(3), self.reflectance(4)
        ndvi = self._divide_zero((nir - red), (nir + red), nan)

        return ndvi