def find_value(self, key):
        """Find a value and return it"""
        values = self.values
        if key not in values:
            raise AttributeError("Config has no value for {}".format(key))

        val = values[key]
        if isinstance(val, Default):
            return val.val
        else:
            return val