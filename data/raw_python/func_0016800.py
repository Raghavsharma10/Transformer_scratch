def scanned(self):
        """Number of items that DynamoDB evaluated, before any filter was applied."""
        if self.request["Select"] == "COUNT":
            while not self.exhausted:
                next(self, None)
        return self._scanned