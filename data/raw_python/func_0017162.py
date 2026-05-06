def _strptime(self, time_str):
        """Convert an ISO 8601 formatted string in UTC into a
        timezone-aware datetime object."""
        if time_str:
            # Parse UTC string into naive datetime, then add timezone
            dt = datetime.strptime(time_str, __timeformat__)
            return dt.replace(tzinfo=UTC())
        return None