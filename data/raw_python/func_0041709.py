def split(self):
        """Immediately stop the current interval and start a new interval that
        has a start_instant equivalent to the stop_interval of self"""
        self.stop()
        interval = Interval()
        interval._start_instant = self.stop_instant
        return interval