def interpolate_grid(self, in_lon, in_lat):
        """
        Interpolates MRMS data to a different grid using cubic bivariate splines
        """
        out_data = np.zeros((self.data.shape[0], in_lon.shape[0], in_lon.shape[1]))
        for d in range(self.data.shape[0]):
            print("Loading ", d, self.variable, self.start_date)
            if self.data[d].max() > -999:
                step = self.data[d]
                step[step < 0] = 0
                if self.lat[-1] < self.lat[0]:
                    spline = RectBivariateSpline(self.lat[::-1], self.lon, step[::-1], kx=3, ky=3)
                else:
                    spline = RectBivariateSpline(self.lat, self.lon, step, kx=3, ky=3)
                print("Evaluating", d, self.variable, self.start_date)
                flat_data = spline.ev(in_lat.ravel(), in_lon.ravel())
                out_data[d] = flat_data.reshape(in_lon.shape)
                del spline
            else:
                print(d, " is missing")
                out_data[d] = -9999
        return out_data