def get_stream_if_exists(self, stream_id):
        """Return a reference to a stream with the given ``stream_id`` if it exists

        This works similar to :py:meth:`get_stream` but will return None if the
        stream is not already created.

        :param stream_id: The path of the stream on Device Cloud
        :raises TypeError: if the stream_id provided is the wrong type
        :raises ValueError: if the stream_id is not properly formed
        :return: :class:`.DataStream` instance with the provided stream_id
        :rtype: :class:`~DataStream`

        """
        stream = self.get_stream(stream_id)
        try:
            stream.get_data_type(use_cached=True)
        except NoSuchStreamException:
            return None
        else:
            return stream