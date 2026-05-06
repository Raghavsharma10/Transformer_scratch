def bulk_write_datapoints(self, datapoints):
        """Perform a bulk write (or set of writes) of a collection of data points

        This method takes a list (or other iterable) of datapoints and writes them
        to Device Cloud in an efficient manner, minimizing the number of HTTP
        requests that need to be made.

        As this call is performed from outside the context of any particular stream,
        each DataPoint object passed in must include information about the stream
        into which the point should be written.

        If all data points being written are for the same stream, you may want to
        consider using :meth:`~DataStream.bulk_write_datapoints` instead.

        Example::

            datapoints = []
            for i in range(300):
                datapoints.append(DataPoint(
                    stream_id="my/stream%d" % (i % 3),
                    data_type=STREAM_TYPE_INTEGER,
                    units="meters",
                    data=i,
                ))
            dc.streams.bulk_write_datapoints(datapoints)

        Depending on the size of the list of datapoints provided, this method may
        need to make multiple calls to Device Cloud (in chunks of 250).

        :param list datapoints: a list of datapoints to be written to Device Cloud
        :raises TypeError: if a list of datapoints is not provided
        :raises ValueError: if any of the provided data points do not have all required
            information (such as information about the stream)
        :raises DeviceCloudHttpException: in the case of an unexpected error in communicating
            with Device Cloud.

        """
        datapoints = list(datapoints)  # effectively performs validation that we have the right type
        for dp in datapoints:
            if not isinstance(dp, DataPoint):
                raise TypeError("All items in the datapoints list must be DataPoints")
            if dp.get_stream_id() is None:
                raise ValueError("stream_id must be set on all datapoints")

        remaining_datapoints = datapoints
        while remaining_datapoints:
            # take up to 250 points and post them until complete
            this_chunk_of_datapoints = remaining_datapoints[:MAXIMUM_DATAPOINTS_PER_POST]
            remaining_datapoints = remaining_datapoints[MAXIMUM_DATAPOINTS_PER_POST:]

            # Build XML list containing data for all points
            datapoints_out = StringIO()
            datapoints_out.write("<list>")
            for dp in this_chunk_of_datapoints:
                datapoints_out.write(dp.to_xml())
            datapoints_out.write("</list>")

            # And send the HTTP Post
            self._conn.post("/ws/DataPoint", datapoints_out.getvalue())
            logger.info('DataPoint batch of %s datapoints written', len(this_chunk_of_datapoints))