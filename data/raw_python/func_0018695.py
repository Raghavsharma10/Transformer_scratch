def to_string(self, hdr, other):
        """String representation with additional information"""
        result = "%s[%s,%s" % (
                hdr, self.get_type(self.type), self.get_clazz(self.clazz))
        if self.unique:
            result += "-unique,"
        else:
            result += ","
        result += self.name
        if other is not None:
            result += ",%s]" % (other)
        else:
            result += "]"
        return result