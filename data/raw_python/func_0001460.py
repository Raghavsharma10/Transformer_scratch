def info(self):
        """list of tuples with QPImage meta data"""
        info = []
        # meta data
        meta = self.meta
        for key in meta:
            info.append((key, self.meta[key]))
        # background correction
        for imdat in [self._amp, self._pha]:
            info += imdat.info
        return info