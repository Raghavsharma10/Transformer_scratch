def last_datetime(self):
        """Return the time of the last operation on the bundle as a datetime object"""
        from datetime import datetime

        try:
            return datetime.fromtimestamp(self.state.lasttime)
        except TypeError:
            return None