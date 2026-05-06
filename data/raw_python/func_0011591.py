def field_datetime_from_json(self, json_val):
        """Convert a UTC timestamp to a UTC datetime."""
        if type(json_val) == int:
            seconds = int(json_val)
            dt = datetime.fromtimestamp(seconds, utc)
        elif json_val is None:
            dt = None
        else:
            seconds, microseconds = [int(x) for x in json_val.split('.')]
            dt = datetime.fromtimestamp(seconds, utc)
            dt += timedelta(microseconds=microseconds)
        return dt