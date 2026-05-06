def distance(self) -> ErrorValue:
        """Sample-to-detector distance"""
        if 'DistCalibrated' in self._data:
            dist = self._data['DistCalibrated']
        else:
            dist = self._data["Dist"]
        if 'DistCalibratedError' in self._data:
            disterr = self._data['DistCalibratedError']
        elif 'DistError' in self._data:
            disterr = self._data['DistError']
        else:
            disterr = 0.0
        return ErrorValue(dist, disterr)