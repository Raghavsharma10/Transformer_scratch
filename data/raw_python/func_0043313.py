def reset(self, period=None):
        """
        Reset the internal timer, effectively causing the next tick to happen
        in `self.period` seconds.

        :param period: If not `None`, specifies a new period to use.
        """
        if period is not None:
            self.period = period

        self.reset_event.set()