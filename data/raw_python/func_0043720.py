def to_timedelta(self):
        """Construct a :class:`timedelta` object from an :class:`nptime`.  The
        timedelta gives the number of seconds (and microseconds) since
        midnight."""
        return timedelta(hours=self.hour, minutes=self.minute,
                seconds=self.second, microseconds=self.microsecond)