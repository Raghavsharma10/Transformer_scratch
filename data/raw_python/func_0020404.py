def __set(self, key, real_value, coded_value):
        """Private method for setting a cookie's value"""
        morse_set = self.get(key, StringMorsel())
        morse_set.set(key, real_value, coded_value)
        dict.__setitem__(self, key, morse_set)