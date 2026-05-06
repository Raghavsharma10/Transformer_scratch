def _dstationarystate(self, k, param):
        """Returns the dstationarystate ."""
        if self._distributionmodel:
            return self.model.dstationarystate(k, param)
        else:
            return self.model.dstationarystate(param)