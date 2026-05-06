def count(self):
        """Number of items that have been loaded from DynamoDB so far, including buffered items."""
        if self.request["Select"] == "COUNT":
            while not self.exhausted:
                next(self, None)
        return self._count