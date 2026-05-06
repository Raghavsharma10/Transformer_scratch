def interp_value(self, lat, lon, indexed=False):
        """ Lookup a pixel value in the raster data, performing linear interpolation
        if necessary. Indexed ==> nearest neighbor (*fast*). """
        (px, py) = self.grid_coordinates.projection_to_raster_coords(lat, lon)
        if indexed:
            return self.raster_data[round(py), round(px)]
        else:
#             from scipy.interpolate import interp2d
#             f_interp = interp2d(self.grid_coordinates.x_axis, self.grid_coordinates.y_axis, self.raster_data, bounds_error=True)
#             return f_interp(lon, lat)[0]
            from scipy.ndimage import map_coordinates
            ret = map_coordinates(self.raster_data, [[py], [px]], order=1)  # linear interp
            return ret[0]