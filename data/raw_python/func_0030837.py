def new_datetime(self):
        """Return the time the bundle was created as a datetime object"""
        from datetime import datetime

        try:
            return datetime.fromtimestamp(self.state.new)
        except TypeError:
            return None