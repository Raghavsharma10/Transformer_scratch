def fcor(self):
        '''
        The CBV-corrected de-trended flux.

        '''

        if self.XCBV is None:
            return None
        else:
            return self.flux - self._mission.FitCBVs(self)