def _get_range_dimension_key(self,
                                 base_key: Key,
                                 start_time: datetime,
                                 end_time: datetime,
                                 count: int = 0) -> List[Tuple[Key, Any]]:
        """
        Returns the list of items from the store based on the given time range or count.

        This is used when the key being used is a DIMENSION key.
        """
        raise NotImplementedError()