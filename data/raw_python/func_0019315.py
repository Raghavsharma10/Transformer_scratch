def series(self) -> InfoArray:
        """Internal time series data within an |numpy.ndarray|."""
        if self.diskflag:
            array = self._load_int()
        elif self.ramflag:
            array = self.__get_array()
        else:
            raise AttributeError(
                f'Sequence {objecttools.devicephrase(self)} is not requested '
                f'to make any internal data available to the user.')
        return InfoArray(array, info={'type': 'unmodified'})