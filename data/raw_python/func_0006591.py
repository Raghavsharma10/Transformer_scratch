def dict(self, name, key_caps=False, value_caps=False):
        '''
        Returns a JSON dict

        @key_caps: Converts all dictionary keys to uppercase
        @value_caps: Converts all dictionary values to uppercase

        @return: JSON item (may be a variable, list or dictionary)
        '''
        # Invalid Dictionary
        if not isinstance(self.json_data[name], dict):
            raise InvalidDictionaryException

        # Convert key and/or values of dictionary to uppercase
        output = {}
        for key, value in self.json_data[name].items():
            output[key.upper() if key_caps else key] = value.upper() if value_caps else value

        return output