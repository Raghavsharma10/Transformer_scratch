def tick(self):
        """Decay the current rate according to the elapsed time"""

        instant_rate = float(self.value) / float(self.tick_interval)

        with self.lock:
            if self.initialized:
                self.rate += (self.alpha * (instant_rate - self.rate))
            else:
                self.initialized = True
                self.rate = instant_rate

            self.value = 0