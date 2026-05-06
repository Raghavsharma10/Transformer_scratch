def normalize_field(self, value):
        """
        Method that must transform the value from string
        Ex: if the expected type is int, it should return int(self._attr)

        """
        if self.default is not None:
            if value is None or value == '':
                value = self.default
        return value