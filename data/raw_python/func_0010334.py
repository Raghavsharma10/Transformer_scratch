def delete_datapoints_in_time_range(self, start_dt=None, end_dt=None):
        """Delete datapoints from this stream between the provided start and end times

        If neither a start or end time is specified, all data points in the stream
        will be deleted.

        :param start_dt: The datetime after which data points should be deleted or None
            if all data points from the beginning of time should be deleted.
        :param end_dt: The datetime before which data points should be deleted or None
            if all data points until the current time should be deleted.
        :raises devicecloud.DeviceCloudHttpException: in the case of an unexpected http error

        """
        start_dt = to_none_or_dt(validate_type(start_dt, datetime.datetime, type(None)))
        end_dt = to_none_or_dt(validate_type(end_dt, datetime.datetime, type(None)))

        params = {}
        if start_dt is not None:
            params['startTime'] = isoformat(start_dt)
        if end_dt is not None:
            params['endTime'] = isoformat(end_dt)

        self._conn.delete("/ws/DataPoint/{stream_id}{querystring}".format(
            stream_id=self.get_stream_id(),
            querystring="?" + urllib.parse.urlencode(params) if params else "",
        ))