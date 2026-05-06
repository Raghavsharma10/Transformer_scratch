def _get_x_axis(self):
        """See http://www.gdal.org/gdal_datamodel.html for details."""
        # 0,0 is top/left top top/left pixel. Actual x/y coord of that pixel are (.5,.5).
        x_centers = np.linspace(.5, self.x_size - .5, self.x_size)
        y_centers = x_centers * 0
        return (self.geotransform[0]
                + self.geotransform[1] * x_centers
                + self.geotransform[2] * y_centers)