def store_extra_info(self, key: str, value: Any) -> None:
        """
        Store some extra value in the messaging storage.

        :param key: key of dictionary entry to add.
        :param value: value of dictionary entry to add.
        :returns: None
        """
        self.extra_keys[key] = value