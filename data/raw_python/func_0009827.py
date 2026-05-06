def format_timestamp(ts):
        """
        Format the UTC timestamp for Elasticsearch
        eg. 2014-07-09T08:37:18.000Z
        @see https://docs.python.org/2/library/time.html#time.strftime
        """
        tz_info = tz.tzutc()
        return datetime.fromtimestamp(ts, tz=tz_info).strftime("%Y-%m-%dT%H:%M:%S.000Z")