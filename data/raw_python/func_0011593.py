def field_timedelta_from_json(self, json_val):
        """Convert json_val to a timedelta object.

        json_val contains total number of seconds in the timedelta.
        If json_val is a string it will be converted to a float.
        """
        if isinstance(json_val, str):
            return timedelta(seconds=float(json_val))
        elif json_val is None:
            return None
        else:
            return timedelta(seconds=json_val)