def notify(self, value):
        """Add a new observation to the metric"""

        with self.lock:
            #TODO: this could slow down slow-rate incoming updates
            # since the number of ticks depends on the actual time
            # passed since the latest notification. Consider using
            # a real timer to tick the EWMA.

            self.tick()

            for avg in (self.m1, self.m5, self.m15, self.day):
                avg.update(value)
            self.count += value