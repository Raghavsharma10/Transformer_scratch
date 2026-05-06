def get_range(self,
                  base_key: Key,
                  start_time: datetime,
                  end_time: datetime = None,
                  count: int = 0) -> List[Tuple[Key, Any]]:
        """
        Returns the list of items from the store based on the given time range or count.
        :param base_key: Items which don't start with the base_key are filtered out.
        :param start_time: Start time to for the range query
        :param end_time: End time of the range query. If None count is used.
        :param count: The number of items to be returned. Used if end_time is not specified.
        """
        if end_time and count:
            raise ValueError('Only one of `end` or `count` can be set')

        if count:
            end_time = datetime.min.replace(
                tzinfo=timezone.utc) if count < 0 else datetime.max.replace(tzinfo=timezone.utc)

        end_time = self._add_timezone_if_required(end_time)
        start_time = self._add_timezone_if_required(start_time)

        if end_time < start_time:
            start_time, end_time = end_time, start_time

        if base_key.key_type == KeyType.TIMESTAMP:
            start_key = Key(KeyType.TIMESTAMP, base_key.identity, base_key.group, [], start_time)
            end_key = Key(KeyType.TIMESTAMP, base_key.identity, base_key.group, [], end_time)
            return self._get_range_timestamp_key(start_key, end_key, count)
        else:
            return self._get_range_dimension_key(base_key, start_time, end_time, count)