def _deserialize(self, data):
        """
        Deserialise from JSON response data.

        String items named ``*_at`` are turned into dates.

        Filters out:
        * attribute names in ``Meta.deserialize_skip``

        :param data dict: JSON-style object with instance data.
        :return: this instance
        """
        if not isinstance(data, dict):
            raise ValueError("Need to deserialize from a dict")

        try:
            skip = set(getattr(self._meta, 'deserialize_skip', []))
        except AttributeError:  # _meta not available
            skip = []

        for key, value in data.items():
            if key not in skip:
                value = self._deserialize_value(key, value)
                setattr(self, key, value)
        return self