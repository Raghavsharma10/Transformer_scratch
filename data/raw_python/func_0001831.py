def combine(self, a, b):
        """A generator that combines two iterables."""

        for l in (a, b):
            for x in l:
                yield x