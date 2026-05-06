def absolute(self):
        """Calculate the absolute value element-wise.

        Returns:
          absolute (Timeseries):
            Absolute value. For complex input (a + b*j) gives sqrt(a**a + b**2)
        """
        da = distob.vectorize(np.absolute)(self)
        return _dts_from_da(da, self.tspan, self.labels)