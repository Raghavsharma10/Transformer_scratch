def built_datetime(self):
        """Return the built time as a datetime object"""
        from datetime import datetime
        try:
            return datetime.fromtimestamp(self.state.build_done)
        except TypeError:
            # build_done is null
            return None