def infos(self, typ, light=None, date=None):
        '''
        Args:
            typ: type of calibration to look for. See .coeffs.keys() for all types available
            date (Optional[str]): date of calibration

        Returns:
            list: all infos available for given typ
        '''
        d = self._getDate(typ, light)
        if date is None:
            return [c[1] for c in d]
        # TODO: not struct time, but time in ms since epoch
        return _getFromDate(d, date)[1]