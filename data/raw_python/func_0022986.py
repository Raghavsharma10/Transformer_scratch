def set_data_values(self, label, x, y, z):
        """
        Set the position of the datapoints
        """
        # TODO: avoid re-allocating an array every time
        self.layers[label]['data'] = np.array([x, y, z]).transpose()
        self._update()