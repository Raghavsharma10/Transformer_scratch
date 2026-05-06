def set_stream_id(self, stream_id):
        """Set the stream id associated with this data point"""
        stream_id = validate_type(stream_id, type(None), *six.string_types)
        if stream_id is not None:
            stream_id = stream_id.lstrip('/')
        self._stream_id = stream_id