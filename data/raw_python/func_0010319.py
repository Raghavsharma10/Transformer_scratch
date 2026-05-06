def from_rollup_json(cls, stream, json_data):
        """Rollup json data from the server looks slightly different

        :param DataStream stream: The :class:`~DataStream` out of which this data is coming
        :param dict json_data: Deserialized JSON data from Device Cloud about this device
        :raises ValueError: if the data is malformed
        :return: (:class:`~DataPoint`) newly created :class:`~DataPoint`
        """
        dp = cls.from_json(stream, json_data)

        # Special handling for timestamp
        timestamp = isoformat(dc_utc_timestamp_to_dt(int(json_data.get("timestamp"))))

        # Special handling for data, all rollup data is float type
        type_converter = _get_decoder_method(stream.get_data_type())
        data = type_converter(float(json_data.get("data")))

        # Update the special fields
        dp.set_timestamp(timestamp)
        dp.set_data(data)
        return dp