def get_data(self, contract=None):
        """Return collected data."""
        if contract is None:
            return self._data
        if contract in self._data.keys():
            return {contract: self._data[contract]}
        raise PyHydroQuebecError("Contract {} not found".format(contract))