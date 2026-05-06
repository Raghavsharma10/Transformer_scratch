def timezone(self, value):
        """Set the timezone."""
        self._timezone = (value if isinstance(value, datetime.tzinfo)
                          else tz.gettz(value))