def timeseries(self):
        """Simulated time series"""
        if self._timeseries is None:
            self.compute()
        if isinstance(self.system, NetworkModel):
            return self.system._reshape_timeseries(self._timeseries)
        else:
            return self._timeseries