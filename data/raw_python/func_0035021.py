def _stationarystate(self, k):
        """Returns the stationarystate ."""
        if self._distributionmodel:
            return self.model.stationarystate(k)
        else:
            return self.model.stationarystate