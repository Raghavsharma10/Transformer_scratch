def _get_y_axis(self):
        """See http://www.gdal.org/gdal_datamodel.html for details."""
        # 0,0 is top/left top top/left pixel. Actual x/y coord of that pixel are (.5,.5).
        y_centers = np.linspace(.5, self.y_size - .5, self.y_size)
        x_centers = y_centers * 0
        return (self.geotransform[3]
                + self.geotransform[4] * x_centers
                + self.geotransform[5] * y_centers)