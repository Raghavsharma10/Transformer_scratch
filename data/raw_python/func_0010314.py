def _get_streams(self, uri_suffix=None):
        """Clear and update internal cache of stream objects"""
        # TODO: handle paging, perhaps change this to be a generator
        if uri_suffix is not None and not uri_suffix.startswith('/'):
            uri_suffix = '/' + uri_suffix
        elif uri_suffix is None:
            uri_suffix = ""
        streams = {}
        response = self._conn.get_json("/ws/DataStream{}".format(uri_suffix))
        for stream_data in response["items"]:
            stream_id = stream_data["streamId"]
            stream = DataStream(self._conn, stream_id, stream_data)
            streams[stream_id] = stream
        return streams