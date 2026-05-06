def request_timestamp(self):
        """
        The timestamp of the request in ISO8601 YYYYMMDD'T'HHMMSS'Z' format.

        If this is not available in the query parameters or headers, or the
        value is not a valid format for AWS SigV4, an AttributeError exception
        is raised.
        """
        amz_date = self.query_parameters.get(_x_amz_date)
        if amz_date is not None:
            amz_date = amz_date[0]
        else:
            amz_date = self.headers.get(_x_amz_date)
            if amz_date is None:
                date = self.headers.get(_date)
                if date is None:
                    raise AttributeError("Date was not passed in the request")

                # This isn't really valid -- seems to be a bug in the AWS
                # documentation.
                if _iso8601_timestamp_regex.match(date):
                    amz_date = date # pragma: nocover
                else:
                    # Parse this as an HTTP date and reformulate it.
                    amz_date = (datetime.strptime(date, _http_date_format)
                                .strftime("%Y%m%dT%H%M%SZ"))
        if not _iso8601_timestamp_regex.match(amz_date):
            raise AttributeError("X-Amz-Date parameter is not a valid ISO8601 "
                                 "string: %r" % amz_date)

        return amz_date