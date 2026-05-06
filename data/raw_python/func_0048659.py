def add(self, key, value):
        # type: (Hashable, Any) -> None
        """
        Adds a new value for the key.

        :param key: the key for the value.
        :param value: the value to add.

        """
        dict.setdefault(self, key, []).append(value)