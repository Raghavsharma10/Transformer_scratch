def get_current_value(self, use_cached=False):
        """Return the most recent DataPoint value written to a stream

        The current value is the last recorded data point for this stream.

        :param bool use_cached: If False, the function will always request the latest from Device Cloud.
            If True, the device will not make a request if it already has cached data.
        :raises devicecloud.DeviceCloudHttpException: in the case of an unexpected http error
        :raises devicecloud.streams.NoSuchStreamException: if this stream has not yet been created
        :return: The most recent value written to this stream (or None if nothing has been written)
        :rtype: :class:`~DataPoint` or None

        """
        current_value = self._get_stream_metadata(use_cached).get("currentValue")
        if current_value:
            return DataPoint.from_json(self, current_value)
        else:
            return None