def dates(self, typ, light=None):
        '''
        Args:
            typ: type of calibration to look for. See .coeffs.keys() for all types available
            light (Optional[str]): restrict to calibrations, done given light source

        Returns:
            list: All calibration dates available for given typ
        '''
        try:
            d = self._getDate(typ, light)
            return [self._toDateStr(c[0]) for c in d]
        except KeyError:
            return []