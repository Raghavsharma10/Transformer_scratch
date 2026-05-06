def beamcenterx(self) -> ErrorValue:
        """X (column) coordinate of the beam center, pixel units, 0-based."""
        try:
            return ErrorValue(self._data['geometry']['beamposy'],
                              self._data['geometry']['beamposy.err'])
        except KeyError:
            return ErrorValue(self._data['geometry']['beamposy'],
                              0.0)