def _get_stream_metadata(self, use_cached):
        """Retrieve metadata about this stream from Device Cloud"""
        if self._cached_data is None or not use_cached:
            try:
                self._cached_data = self._conn.get_json("/ws/DataStream/%s" % self._stream_id)["items"][0]
            except DeviceCloudHttpException as http_exception:
                if http_exception.response.status_code == 404:
                    raise NoSuchStreamException("Stream with id %r has not been created" % self._stream_id)
                raise http_exception
        return self._cached_data