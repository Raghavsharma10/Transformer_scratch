def get_data_ttl(self, use_cached=True):
        """Retrieve the dataTTL for this stream

        The dataTtl is the time to live (TTL) in seconds for data points stored in the data stream.
        A data point expires after the configured amount of time and is automatically deleted.

        :param bool use_cached: If False, the function will always request the latest from Device Cloud.
            If True, the device will not make a request if it already has cached data.
        :raises devicecloud.DeviceCloudHttpException: in the case of an unexpected http error
        :raises devicecloud.streams.NoSuchStreamException: if this stream has not yet been created
        :return: The dataTtl associated with this stream in seconds
        :rtype: int or None

        """

        data_ttl_text = self._get_stream_metadata(use_cached).get("dataTtl")
        return int(data_ttl_text)