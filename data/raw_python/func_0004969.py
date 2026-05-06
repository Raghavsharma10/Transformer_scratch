def date(self) -> datetime.datetime:
        """Date of the experiment (start of exposure)"""
        return self._data['Date'] - datetime.timedelta(0, float(self.exposuretime), 0)