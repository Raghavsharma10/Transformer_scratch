def get_rollup_ttl(self, use_cached=True):
        """Retrieve the rollupTtl for this stream

        The rollupTtl is the time to live (TTL) in seconds for the aggregate roll-ups of data points
        stored in the stream. A roll-up expires after the configured amount of time and is
        automatically deleted.

        :param bool use_cached: If False, the function will always request the latest from Device Cloud.
            If True, the device will not make a request if it already has cached data.
        :raises devicecloud.DeviceCloudHttpException: in the case of an unexpected http error
        :raises devicecloud.streams.NoSuchStreamException: if this stream has not yet been created
        :return: The rollupTtl associated with this stream in seconds
        :rtype: int or None

        """
        rollup_ttl_text = self._get_stream_metadata(use_cached).get("rollupTtl")
        return int(rollup_ttl_text)