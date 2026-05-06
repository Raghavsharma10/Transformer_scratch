def build_duration_pretty(self):
        """Return the difference between build and build_done states, in a human readable format"""
        from ambry.util import pretty_time
        from time import time

        if not self.state.building:
            return None

        built = self.state.built or time()

        try:
            return pretty_time(int(built) - int(self.state.building))
        except TypeError:  # one of the values is  None or not a number
            return None