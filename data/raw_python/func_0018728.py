def entries(self):
        """Returns a list of all entries"""
        def add(x, y):
            return x + y
        try:
            return reduce(add, list(self.cache.values()))
        except:
            return []