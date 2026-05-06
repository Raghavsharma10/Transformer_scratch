def add(self, entry):
        """Adds an entry"""
        if self.get(entry) is not None:
            return
        try:
            list = self.cache[entry.key]
        except:
            list = self.cache[entry.key] = []
        list.append(entry)