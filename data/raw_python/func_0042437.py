def type(self):
        """Returns 'number', 'string', 'date' or 'unknown' based on the type of the value"""
        if isinstance(self.value, numbers.Number):
            return "number"
        if isinstance(self.value, basestring):
            return "string"
        return "unknown"