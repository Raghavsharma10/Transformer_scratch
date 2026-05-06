def projection_to_raster_coords(self, lat, lon):
        """ Returns pixel centers.
        See documentation for the GDAL function GetGeoTransform for details. """
        r_px_py = np.array([1, lon, lat])
        tg = inv(np.array([[1, 0, 0], self.geotransform[0:3], self.geotransform[3:6]]))
        return np.inner(tg, r_px_py)[1:]