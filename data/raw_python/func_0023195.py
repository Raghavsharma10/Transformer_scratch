def reserve(self, capacity):
        """ Set current capacity of the underlying array"""

        if capacity >= self._data.size:
            capacity = int(2 ** np.ceil(np.log2(capacity)))
            self._data = np.resize(self._data, capacity)