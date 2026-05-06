def read(self, start_time=None, end_time=None, use_client_timeline=True, newest_first=True,
             rollup_interval=None, rollup_method=None, timezone=None, page_size=1000):
        """Read one or more DataPoints from a stream

        .. warning::
           The data points from Device Cloud is a paged data set.  When iterating over the
           result set there could be delays when we hit the end of a page.  If this is undesirable,
           the caller should collect all results into a data structure first before iterating over
           the result set.

        :param start_time: The start time for the window of data points to read.  None means
            that we should start with the oldest data available.
        :type start_time: :class:`datetime.datetime` or None
        :param end_time: The end time for the window of data points to read.  None means
            that we should include all points received until this point in time.
        :type end_time: :class:`datetime.datetime` or None
        :param bool use_client_timeline: If True, the times used will be those provided by
              clients writing data points into the cloud (which also default to server time
              if the a timestamp was not included by the client).  This is usually what you
              want.  If False, the server timestamp will be used which records when the data
              point was received.
        :param bool newest_first: If True, results will be ordered from newest to oldest (descending order).
            If False, results will be returned oldest to newest.
        :param rollup_interval: the roll-up interval that should be used if one is desired at all.  Rollups
            will not be performed if None is specified for the interval.  Valid roll-up interval values
            are None, "half", "hourly", "day", "week", and "month".  See `DataPoints documentation
            <http://ftp1.digi.com/support/documentation/html/90002008/90002008_P/Default.htm#ProgrammingTopics/DataStreams.htm#DataPoints>`_
            for additional details on these values.
        :type rollup_interval: str or None
        :param rollup_method: The aggregation applied to values in the points within the specified
            rollup_interval.  Available methods are None, "sum", "average", "min", "max", "count", and
            "standarddev".  See `DataPoint documentation
            <http://ftp1.digi.com/support/documentation/html/90002008/90002008_P/Default.htm#ProgrammingTopics/DataStreams.htm#DataPoints>`_
            for additional details on these values.
        :type rollup_method: str or None
        :param timezone: timezone for calculating roll-ups. This determines roll-up interval
            boundaries and only applies to roll-ups of a day or larger (for example, day,
            week, or month). Note that it does not apply to the startTime and endTime parameters.
            See the `Timestamps <http://ftp1.digi.com/support/documentation/html/90002008/90002008_P/Default.htm#ProgrammingTopics/DataStreams.htm#timestamp>`_
            and `Supported Time Zones <http://ftp1.digi.com/support/documentation/html/90002008/90002008_P/Default.htm#ProgrammingTopics/DataStreams.htm#TimeZones>`_
            sections for more information.
        :type timezone: str or None
        :param int page_size: The number of results that we should attempt to retrieve from the
            device cloud in each page.  Generally, this can be left at its default value unless
            you have a good reason to change the parameter for performance reasons.
        :returns: A generator object which one can iterate over the DataPoints read.

        """

        is_rollup = False
        if (rollup_interval is not None) or (rollup_method is not None):
            is_rollup = True
            numeric_types = [
                STREAM_TYPE_INTEGER,
                STREAM_TYPE_LONG,
                STREAM_TYPE_FLOAT,
                STREAM_TYPE_DOUBLE,
                STREAM_TYPE_STRING,
                STREAM_TYPE_BINARY,
                STREAM_TYPE_UNKNOWN,
            ]

            if self.get_data_type(use_cached=True) not in numeric_types:
                raise InvalidRollupDatatype('Rollups only support numerical DataPoints')

        # Validate function inputs
        start_time = to_none_or_dt(validate_type(start_time, datetime.datetime, type(None)))
        end_time = to_none_or_dt(validate_type(end_time, datetime.datetime, type(None)))
        use_client_timeline = validate_type(use_client_timeline, bool)
        newest_first = validate_type(newest_first, bool)
        rollup_interval = validate_type(rollup_interval, type(None), *six.string_types)
        if not rollup_interval in {None,
                                   ROLLUP_INTERVAL_HALF,
                                   ROLLUP_INTERVAL_HOUR,
                                   ROLLUP_INTERVAL_DAY,
                                   ROLLUP_INTERVAL_WEEK,
                                   ROLLUP_INTERVAL_MONTH, }:
            raise ValueError("Invalid rollup_interval %r provided" % (rollup_interval, ))
        rollup_method = validate_type(rollup_method, type(None), *six.string_types)
        if not rollup_method in {None,
                                 ROLLUP_METHOD_SUM,
                                 ROLLUP_METHOD_AVERAGE,
                                 ROLLUP_METHOD_MIN,
                                 ROLLUP_METHOD_MAX,
                                 ROLLUP_METHOD_COUNT,
                                 ROLLUP_METHOD_STDDEV}:
            raise ValueError("Invalid rollup_method %r provided" % (rollup_method, ))
        timezone = validate_type(timezone, type(None), *six.string_types)
        page_size = validate_type(page_size, *six.integer_types)

        # Remember that there could be multiple pages of data and we want to provide
        # in iterator over the result set.  To start the process out, we need to make
        # an initial request without a page cursor.  We should get one in response to
        # our first request which we will use to page through the result set
        query_parameters = {
            'timeline': 'client' if use_client_timeline else 'server',
            'order': 'descending' if newest_first else 'ascending',
            'size': page_size
        }
        if start_time is not None:
            query_parameters["startTime"] = isoformat(start_time)
        if end_time is not None:
            query_parameters["endTime"] = isoformat(end_time)
        if rollup_interval is not None:
            query_parameters["rollupInterval"] = rollup_interval
        if rollup_method is not None:
            query_parameters["rollupMethod"] = rollup_method
        if timezone is not None:
            query_parameters["timezone"] = timezone

        result_size = page_size
        while result_size == page_size:
            # request the next page of data or first if pageCursor is not set as query param
            try:
                result = self._conn.get_json("/ws/DataPoint/{stream_id}?{query_params}".format(
                    stream_id=self.get_stream_id(),
                    query_params=urllib.parse.urlencode(query_parameters)
                ))
            except DeviceCloudHttpException as http_exception:
                if http_exception.response.status_code == 404:
                    raise NoSuchStreamException()
                raise http_exception

            result_size = int(result["resultSize"])  # how many are actually included here?
            query_parameters["pageCursor"] = result.get("pageCursor")  # will not be present if result set is empty
            for item_info in result.get("items", []):
                if is_rollup:
                    data_point = DataPoint.from_rollup_json(self, item_info)
                else:
                    data_point = DataPoint.from_json(self, item_info)
                yield data_point