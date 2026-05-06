def build(self):
        """
        Creates the objects from the JSON response.
        """

        if self.json['sys']['type'] == 'Array':
            return self._build_array()
        return self._build_item(self.json)