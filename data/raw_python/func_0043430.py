def import_data(self, data):
        """
        Set the fields established in data to the instance
        """
        if self.get_read_only() and self.is_locked():
            return

        if isinstance(data, BaseModel):
            data = data.export_data()

        if not isinstance(data, (dict, Mapping)):
            raise TypeError('Impossible to import data')

        self._import_data(data)