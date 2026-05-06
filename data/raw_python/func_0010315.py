def create_stream(self, stream_id, data_type, description=None, data_ttl=None,
                      rollup_ttl=None, units=None):
        """Create a new data stream on Device Cloud

        This method will attempt to create a new data stream on Device Cloud.
        This method will only succeed if the stream does not already exist.

        :param str stream_id: The path/id of the stream being created on Device Cloud.
        :param str data_type: The type of this stream.  This must be in the set
            `{ INTEGER, LONG, FLOAT, DOUBLE, STRING, BINARY, UNKNOWN }`.  These values are
            available in constants like :attr:`~STREAM_TYPE_INTEGER`.
        :param str description: An optional description of this stream. See :meth:`~DataStream.get_description`.
        :param int data_ttl: The TTL for data points in this stream. See :meth:`~DataStream.get_data_ttl`.
        :param int rollup_ttl: The TTL for performing rollups on data. See :meth:~DataStream.get_rollup_ttl`.
        :param str units: Units for data in this stream.  See :meth:`~DataStream.get_units`

        """

        stream_id = validate_type(stream_id, *six.string_types)
        data_type = validate_type(data_type, type(None), *six.string_types)
        if isinstance(data_type, *six.string_types):
            data_type = str(data_type).upper()
        if not data_type in (set([None, ]) | set(list(DSTREAM_TYPE_MAP.keys()))):
            raise ValueError("data_type %r is not valid" % data_type)
        description = validate_type(description, type(None), *six.string_types)
        data_ttl = validate_type(data_ttl, type(None), *six.integer_types)
        rollup_ttl = validate_type(rollup_ttl, type(None), *six.integer_types)
        units = validate_type(units, type(None), *six.string_types)

        sio = StringIO()
        sio.write("<DataStream>")
        conditional_write(sio, "<streamId>{}</streamId>", stream_id)
        conditional_write(sio, "<dataType>{}</dataType>", data_type)
        conditional_write(sio, "<description>{}</description>", description)
        conditional_write(sio, "<dataTtl>{}</dataTtl>", data_ttl)
        conditional_write(sio, "<rollupTtl>{}</rollupTtl>", rollup_ttl)
        conditional_write(sio, "<units>{}</units>", units)
        sio.write("</DataStream>")

        self._conn.post("/ws/DataStream", sio.getvalue())
        logger.info("Data stream (%s) created successfully", stream_id)
        stream = DataStream(self._conn, stream_id)
        return stream