def tick(self):
        """
        Emulate a timer: in order to avoid a real timer we "tick" a number
        of times depending on the actual time passed since the last tick
        """

        now = time.time()

        elapsed = now - self.latest_tick

        if elapsed > self.tick_interval:
            ticks = int(elapsed / self.tick_interval)

            self.tick_all(ticks)

            self.latest_tick = now