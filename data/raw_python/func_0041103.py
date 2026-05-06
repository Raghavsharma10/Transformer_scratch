def _rnd_datetime(self, start, end):
        """Internal random datetime generator.
        """
        return self.from_utctimestamp(
            random.randint(
                int(self.to_utctimestamp(start)),
                int(self.to_utctimestamp(end)),
            )
        )