def set_data(self, data):
        """
        Fills form with data

        Args:
            data (dict): Data to assign form fields.

        Returns:
            Self. Form object.

        """
        for name in self._fields:
            setattr(self, name, data.get(name))
        return self