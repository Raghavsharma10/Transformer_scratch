def set_min(self, fmin):
        """
        Updates minimum value
        """
        if round(100000*fmin) != 100000*fmin:
            raise DriverError('utils.widgets.Expose.set_min: ' +
                              'fmin must be a multiple of 0.00001')
        self.fmin = fmin
        self.set(self.fmin)