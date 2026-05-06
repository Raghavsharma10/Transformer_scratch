def get_miles(self):
    ''' convert the measurement to inches '''
    if self._obs_value in self.MISSING:
      return 'MISSING'
    if self._obs_units == self.METERSPERSECOND:
      return round(2.23694 * self._obs_value, 4)