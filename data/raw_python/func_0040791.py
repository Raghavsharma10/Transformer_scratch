def localize_field(self, value):
        """
        Method that must transform the value from object to localized string

        """
        if self.default is not None:
            if value is None or value == '':
                value = self.default
        return value or ''