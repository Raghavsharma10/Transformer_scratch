def data(self, **kwargs):
        """
        If a key is passed in, a corresponding value will be returned.
        If a key-value pair is passed in then the corresponding key in
        the database will be set to the specified value.
        A dictionary can be passed in as well.
        If a key does not exist and a value is provided then an entry
        will be created in the database.
        """

        key = kwargs.pop('key', None)
        value = kwargs.pop('value', None)
        dictionary = kwargs.pop('dictionary', None)

        # Fail if a key and a dictionary or a value and a dictionary are given
        if (key is not None and dictionary is not None) or \
           (value is not None and dictionary is not None):
            raise ValueError

        # If only a key was provided return the corresponding value
        if key is not None and value is None:
            return self._get_content(key)

        # if a key and a value are passed in
        if key is not None and value is not None:
            self._set_content(key, value)

        if dictionary is not None:
            for key in dictionary.keys():
                value = dictionary[key]
                self._set_content(key, value)

        return self._get_content()