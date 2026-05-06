def absolute(self):
        """Calculate the absolute value element-wise.

        Returns:
          absolute (Timeseries):
            Absolute value. For complex input (a + b*j) gives sqrt(a**a + b**2)
        """
        return Timeseries(np.absolute(self), self.tspan, self.labels)