def scan(self, t, dt=None, aggfunc=None):
        """
        Returns the spectrum from a specific time or range of times.
        """
        return self.data.scan(t, dt, aggfunc)