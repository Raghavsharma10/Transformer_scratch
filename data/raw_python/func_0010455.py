def get_inches(self):
    ''' convert the measurement to inches '''
    if self._obs_value in self.MISSING:
      return 'MISSING'
    if self._obs_units == self.MILLIMETERS:
      return round(self.INCH_CONVERSION_FACTOR * self._obs_value, 4)