def from_json(cls, stream, json_data):
        """Create a new DataPoint object from device cloud JSON data

        :param DataStream stream: The :class:`~DataStream` out of which this data is coming
        :param dict json_data: Deserialized JSON data from Device Cloud about this device
        :raises ValueError: if the data is malformed
        :return: (:class:`~DataPoint`) newly created :class:`~DataPoint`

        """
        type_converter = _get_decoder_method(stream.get_data_type())
        data = type_converter(json_data.get("data"))
        return cls(
            # these are actually properties of the stream, not the data point
            stream_id=stream.get_stream_id(),
            data_type=stream.get_data_type(),
            units=stream.get_units(),

            # and these are part of the data point itself
            data=data,
            description=json_data.get("description"),
            timestamp=json_data.get("timestampISO"),
            server_timestamp=json_data.get("serverTimestampISO"),
            quality=json_data.get("quality"),
            location=json_data.get("location"),
            dp_id=json_data.get("id"),
        )