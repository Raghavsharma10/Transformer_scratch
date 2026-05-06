def dump(self):
        """
        Dump raw JSON output of matching queryset objects.

        Returns:
            List of dicts.

        """
        results = []
        for data in self.data():
            results.append(data)
        return results