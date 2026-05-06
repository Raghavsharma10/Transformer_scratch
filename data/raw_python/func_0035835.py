def save_to_json(self):
        """The method saves data to json from object"""
        requestvalues = {
            'Dataset': self.dataset,
            'Header' : self._get_item_array(self.header),
            'Filter' : self._get_item_array(self.filter),
            'Stub' : self._get_item_array(self.stub),
            'Frequencies': self.frequencies
        }
        return json.dumps(requestvalues)