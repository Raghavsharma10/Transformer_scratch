def raster_to_projection_coords(self, pixel_x, pixel_y):
        """ Use pixel centers when appropriate.
        See documentation for the GDAL function GetGeoTransform for details. """
        h_px_py = np.array([1, pixel_x, pixel_y])
        gt = np.array([[1, 0, 0], self.geotransform[0:3], self.geotransform[3:6]])
        arr = np.inner(gt, h_px_py)
        return arr[2], arr[1]