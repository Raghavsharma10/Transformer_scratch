def enforce_type(self, attr, val):
        """converts a value to the attribute's type"""
        if not attr in self.types:
            return utfstr(val)
        elif self.types[attr] == 'int':
            return int(float(val))
        elif self.types[attr] == 'float':
            return float(val)
        else:
            return utfstr(val)