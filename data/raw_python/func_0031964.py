def get(self):
        """
        Return the computed statistics over the gathered data
        """

        with self.lock:
            self.tick()

            data = dict(
                kind="meter",
                count=self.count,
                mean=self.count / (time.time() - self.started_on),
                one=self.m1.rate,
                five=self.m5.rate,
                fifteen=self.m15.rate,
                day=self.day.rate)

        return data