def _attributeStr(self, name):
        """
        Return name=value for a single attribute
        """
        return "{}={}".format(
            _encodeAttr(name),
            ",".join([_encodeAttr(v) for v in self.attributes[name]]))