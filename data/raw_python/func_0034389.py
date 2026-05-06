def _get(self, item):
        """
        Helper function to keep the __getattr__ and __getitem__ calls
        KISSish
        """
        if item not in object.__getattribute__(self, "_protectedItems") \
                and item[0] != "_":
            data = object.__getattribute__(self, "_data")
            if item in data:
                return data[item]
        return object.__getattribute__(self, item)