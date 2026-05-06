def flux(self) -> ErrorValue:
        """X-ray flux in photons/sec."""
        try:
            return ErrorValue(self._data['Flux'], self._data.setdefault('FluxError',0.0))
        except KeyError:
            return 1 / self.pixelsizex / self.pixelsizey / ErrorValue(self._data['NormFactor'],
                                                                      self._data.setdefault('NormFactorError',0.0))