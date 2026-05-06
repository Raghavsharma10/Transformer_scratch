def as_dict(self):
        """
        turns attribute filter object into python dictionary
        """
        output_dictionary = dict()

        for key, value in iter(self._key_map.items()):
            if isinstance(value, bool):
                output_dictionary[key] = value
            elif isinstance(value, self.__class__):
                output_dictionary[key] = value.as_dict()

        return output_dictionary