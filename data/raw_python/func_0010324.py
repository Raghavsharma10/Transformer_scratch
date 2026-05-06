def set_data_type(self, data_type):
        """Set the data type for ths data point

        The data type is actually associated with the stream itself and should
        not (generally) vary on a point-per-point basis.  That being said, if
        creating a new stream by writing a datapoint, it may be beneficial to
        include this information.

        The data type provided should be in the set of available data types of
        { INTEGER, LONG, FLOAT, DOUBLE, STRING, BINARY, UNKNOWN }.

        """
        validate_type(data_type, type(None), *six.string_types)
        if isinstance(data_type, *six.string_types):
            data_type = str(data_type).upper()
        if not data_type in ({None} | set(DSTREAM_TYPE_MAP.keys())):
            raise ValueError("Provided data type not in available set of types")
        self._data_type = data_type