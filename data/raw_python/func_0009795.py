def get_complex_value(self, name):
        """Get a value from the service dictionaries.

        It's best to use get_value if it has the data you require since
        the vera subscription only updates data in dev_info.
        """
        for item in self.json_state.get('states'):
            if item.get('variable') == name:
                return item.get('value')
        return None