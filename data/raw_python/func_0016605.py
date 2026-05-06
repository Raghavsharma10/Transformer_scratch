def add(self, k, count=1):
        """
        Increments the count for the given element.
        """
        self.counts[k] += count
        self.total += count