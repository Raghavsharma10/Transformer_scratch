def _attributeStrs(self):
        """
        Return name=value, semi-colon-separated string for attributes,
        including url-style quoting
        """
        return ";".join([self._attributeStr(name)
                         for name in self.attributes.iterkeys()])