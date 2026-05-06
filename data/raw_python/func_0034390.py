def _set(self, item, value):
        """
        Helper function to keep the __setattr__ and __setitem__ calls
        KISSish

        Will only set the objects _data if the given items name is not prefixed
        with _ or if the item exists in the protected items List.
        """
        if item not in object.__getattribute__(self, "_protectedItems") \
                and item[0] != "_":
            keys = object.__getattribute__(self, "_data")
            if not hasattr(value, '__call__'):
                keys[item] = value
                return value
            if hasattr(value, '__call__') and item in keys:
                raise Exception("""Cannot set model data to a function, same \
name exists in data""")
        return object.__setattr__(self, item, value)