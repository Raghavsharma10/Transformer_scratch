def trim(self, start, stop):
        """Trim the list to the specified range of elements."""
        return self.client.ltrim(self.name, start, stop - 1)