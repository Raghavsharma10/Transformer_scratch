def _get_range_timestamp_key(self, start: Key, end: Key,
                                 count: int = 0) -> List[Tuple[Key, Any]]:
        """
        Returns the list of items from the store based on the given time range or count.

        This is used when the key being used is a TIMESTAMP key.
        """
        raise NotImplementedError()