def write(self, datapoint):
        """Write some raw data to a stream using the DataPoint API

        This method will mutate the datapoint provided to populate it with information
        available from the stream as it is available (but without making any new HTTP
        requests).  For instance, we will add in information about the stream data
        type if it is available so that proper type conversion happens.

        Values already set on the datapoint will not be overridden (except for path)

        :param DataPoint datapoint: The :class:`.DataPoint` that should be written to Device Cloud

        """
        if not isinstance(datapoint, DataPoint):
            raise TypeError("First argument must be a DataPoint object")

        datapoint._stream_id = self.get_stream_id()
        if self._cached_data is not None and datapoint.get_data_type() is None:
            datapoint._data_type = self.get_data_type()

        self._conn.post("/ws/DataPoint/{}".format(self.get_stream_id()), datapoint.to_xml())