def beamcentery(self) -> ErrorValue:
        """Y (row) coordinate of the beam center, pixel units, 0-based."""
        try:
            return ErrorValue(self._data['geometry']['beamposx'],
                              self._data['geometry']['beamposx.err'])
        except KeyError:
            return ErrorValue(self._data['geometry']['beamposx'],
                              0.0)