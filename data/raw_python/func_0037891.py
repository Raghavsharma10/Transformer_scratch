def _is_text_data(self, data_type):
        """Private method for testing text data types."""
        dt = DATA_TYPES[data_type]
        if type(self.data) is dt['type'] and len(self.data) < dt['max'] and all(type(char) == str for char in self.data):
            self.type = data_type.upper()
            self.len = len(self.data)
            return True