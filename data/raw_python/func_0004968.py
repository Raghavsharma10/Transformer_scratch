def temperature(self) -> Optional[ErrorValue]:
        """Sample temperature"""
        try:
            return ErrorValue(self._data['Temperature'], self._data.setdefault('TemperatureError', 0.0))
        except KeyError:
            return None